"""Structure-aware loaders for PDF, Markdown, images, text and DOCX."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from app.config import Settings
from app.core.types import Block, ParsedDocument
from app.core.utils import clean_extracted_text, normalize_text, safe_filename, truncate
from app.ingest.vision import VisionService

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FRONTMATTER_KEY = re.compile(r"^([A-Za-z_][\w-]*):\s*(.*)$")


class HeadingStack:
    def __init__(self) -> None:
        self.items: list[tuple[int, str]] = []

    def update(self, level: int, title: str) -> list[str]:
        title = normalize_text(title)
        if not title:
            return self.current
        self.items = [(lv, tx) for lv, tx in self.items if lv < level]
        self.items.append((level, truncate(title, 80)))
        return self.current

    @property
    def current(self) -> list[str]:
        return [title for _, title in self.items][-4:]


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return path.read_text(encoding="utf-8", errors="replace")


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for idx in range(1, min(len(lines), 80)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        match = _FRONTMATTER_KEY.match(line.strip())
        if match:
            meta[match.group(1).lower()] = match.group(2).strip().strip("\"'")
    return meta, "\n".join(lines[end + 1 :])


def _parse_markdown(
    path: Path, name: str, settings: Settings, vision: VisionService
) -> ParsedDocument:
    raw = _read_text(path)
    frontmatter, text = _split_frontmatter(raw)
    blocks: list[Block] = []
    warnings: list[str] = []
    stack = HeadingStack()
    buffer: list[str] = []
    title = frontmatter.get("title") or ""
    image_count = 0

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        body = clean_extracted_text("\n".join(buffer))
        if body:
            blocks.append(Block(text=body, heading_path=stack.current, content_type="text"))
        buffer = []

    def add_image(alt: str, source: str) -> None:
        nonlocal image_count
        src = source.strip()
        if re.match(r"^[a-z]+://", src, re.I) or src.startswith("data:"):
            warnings.append(f"跳过远程图片: {src[:80]}")
            return
        candidate = (path.parent / src).resolve()
        try:
            inside = candidate.is_relative_to(path.parent.resolve())
        except Exception:
            inside = False
        if not inside or not candidate.exists() or candidate.suffix.lower() not in _IMAGE_SUFFIXES:
            warnings.append(f"未找到本地图片: {src[:80]}")
            return
        image_count += 1
        result = vision.extract(candidate)
        image_text = result.combined_text() or f"【图片】{alt or candidate.stem}"
        blocks.append(
            Block(
                text=image_text,
                heading_path=stack.current,
                content_type="vlm" if result.provider.startswith("vlm") else "image",
                metadata={
                    "image_path": str(candidate.relative_to(path.parent)),
                    "provider": result.provider,
                    "caption": result.caption,
                    "confidence": result.confidence,
                },
            )
        )
        warnings.extend(result.warnings)

    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            stack.update(level, heading.group(2))
            if level == 1 and not title:
                title = heading.group(2).strip()
            idx += 1
            continue

        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            language = stripped[3:].strip()
            code_lines: list[str] = []
            idx += 1
            while idx < len(lines) and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            code_body = "\n".join(code_lines).strip()
            if code_body:
                blocks.append(
                    Block(
                        text=f"```{language}\n{code_body}\n```",
                        heading_path=stack.current,
                        content_type="code",
                        metadata={"language": language},
                    )
                )
            idx += 1
            continue

        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush()
            rows: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                rows.append(lines[idx].strip())
                idx += 1
            table = "\n".join(rows)
            if table:
                blocks.append(
                    Block(text=table, heading_path=stack.current, content_type="table")
                )
            continue

        images = list(_MD_IMAGE.finditer(line))
        if images:
            if line[: images[0].start()].strip():
                buffer.append(line[: images[0].start()])
            flush()
            for match in images:
                alt, src = match.group(1), match.group(2)
                add_image(alt, src)
            tail = line[images[-1].end() :].strip()
            if tail:
                buffer.append(tail)
            idx += 1
            continue

        if not stripped:
            flush()
            idx += 1
            continue
        buffer.append(line)
        idx += 1
    flush()

    if not title:
        title = frontmatter.get("title") or path.stem
    if not blocks:
        warnings.append("文档没有可索引的文本内容")
    return ParsedDocument(
        title=title.strip() or name,
        source_type="markdown",
        blocks=blocks,
        page_count=0,
        metadata={
            "parser": "markdown",
            "frontmatter": frontmatter,
            "images": image_count,
            "chars": sum(len(b.text) for b in blocks),
        },
        warnings=warnings,
    )


def _extract_pdf_tables(page) -> list[str]:
    tables: list[str] = []
    try:
        found = page.find_tables()
        for table in getattr(found, "tables", []) or []:
            rows = table.extract() or []
            cleaned_rows: list[str] = []
            for row in rows:
                cells = [normalize_text(str(cell or "")) for cell in row]
                if any(cells):
                    cleaned_rows.append(" | ".join(cells))
            if cleaned_rows:
                tables.append("\n".join(cleaned_rows))
    except Exception:
        pass
    return tables


def _parse_pdf(
    path: Path, name: str, settings: Settings, vision: VisionService, doc_id: str
) -> ParsedDocument:
    blocks: list[Block] = []
    warnings: list[str] = []
    title = ""
    page_count = 0
    image_count = 0
    table_count = 0
    heading_stack = HeadingStack()

    try:
        import pymupdf  # type: ignore
    except Exception:
        pymupdf = None  # type: ignore

    if pymupdf is None:
        # Fallback: pypdf text extraction (no OCR / images).
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean_extracted_text(page.extract_text() or "")
            if text:
                blocks.append(Block(text=text, page=page_no, content_type="text"))
        warnings.append("PyMuPDF 不可用，已回退到 pypdf 纯文本解析")
    else:
        try:
            document = pymupdf.open(str(path))
        except Exception as exc:
            raise ValueError(f"PDF 打不开: {exc}") from exc
        page_count = document.page_count
        try:
            metadata = document.metadata or {}
            title = str(metadata.get("title") or "").strip()
        except Exception:
            metadata = {}
        image_budget = max(0, settings.vlm_max_images_per_doc)
        seen_xrefs: set[int] = set()
        out_dir = settings.extracted_dir / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for page_index in range(page_count):
            page = document[page_index]
            page_no = page_index + 1
            page_text = clean_extracted_text(page.get_text("text") or "")
            page_blocks: list[Block] = []
            try:
                page_dict = page.get_text("dict")
                raw_blocks = []
                sizes: list[float] = []
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
                    if not spans:
                        continue
                    text = normalize_text("".join(span.get("text", "") for span in spans))
                    if not text:
                        continue
                    size = max(float(span.get("size", 0)) for span in spans)
                    raw_blocks.append((text, size))
                    sizes.append(size)
                median = sorted(sizes)[len(sizes) // 2] if sizes else 10.0
                for text, size in raw_blocks:
                    if len(text) <= 80 and size >= median * 1.18:
                        level = 1 if size >= median * 1.45 else 2
                        heading_stack.update(level, text)
                        if not title and level == 1 and len(text) < 60:
                            title = text
                    else:
                        page_blocks.append(
                            Block(
                                text=text,
                                page=page_no,
                                heading_path=heading_stack.current,
                                content_type="text",
                            )
                        )
            except Exception:
                if page_text:
                    page_blocks = [
                        Block(text=page_text, page=page_no, content_type="text")
                    ]
            if not page_blocks and page_text:
                page_blocks.append(
                    Block(text=page_text, page=page_no, heading_path=heading_stack.current)
                )

            for table_text in _extract_pdf_tables(page):
                table_count += 1
                page_blocks.append(
                    Block(
                        text=table_text,
                        page=page_no,
                        heading_path=heading_stack.current,
                        content_type="table",
                    )
                )

            blocks.extend(page_blocks)

            # Scanned / image-heavy page -> render and OCR.
            if len(page_text) < 40 and settings.ocr_enabled:
                try:
                    png_path = out_dir / f"page_{page_no}.png"
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    pix.save(str(png_path))
                    result = vision.extract(png_path)
                    if result.combined_text():
                        blocks.append(
                            Block(
                                text=result.combined_text(),
                                page=page_no,
                                heading_path=heading_stack.current,
                                content_type="ocr" if "ocr" in result.provider else "vlm",
                                metadata={
                                    "provider": result.provider,
                                    "rendered_page": True,
                                    "confidence": result.confidence,
                                },
                            )
                        )
                        warnings.extend(result.warnings)
                except Exception as exc:
                    warnings.append(f"第 {page_no} 页 OCR 失败: {exc}")

            # Embedded figures (captions are useful semantic content).
            if image_budget > 0:
                try:
                    for image in page.get_images(full=True):
                        if image_count >= image_budget:
                            break
                        xref = int(image[0])
                        if xref in seen_xrefs:
                            continue
                        seen_xrefs.add(xref)
                        info = document.extract_image(xref)
                        width = int(info.get("width") or 0)
                        height = int(info.get("height") or 0)
                        if width < 80 or height < 80:
                            continue
                        ext = info.get("ext") or "png"
                        image_path = out_dir / f"p{page_no}_x{xref}.{ext}"
                        image_path.write_bytes(info["image"])
                        image_count += 1
                        result = vision.extract(image_path)
                        if result.combined_text():
                            blocks.append(
                                Block(
                                    text=result.combined_text(),
                                    page=page_no,
                                    heading_path=heading_stack.current,
                                    content_type="vlm" if result.provider.startswith("vlm") else "image",
                                    metadata={
                                        "provider": result.provider,
                                        "image_width": width,
                                        "image_height": height,
                                        "caption": result.caption,
                                    },
                                )
                            )
                            warnings.extend(result.warnings)
                except Exception as exc:
                    warnings.append(f"第 {page_no} 页图片解析失败: {exc}")
        document.close()

    if not title:
        for block in blocks:
            first_line = block.text.splitlines()[0].strip() if block.text else ""
            if 3 <= len(first_line) <= 60:
                title = first_line
                break
    if not blocks:
        warnings.append("PDF 未提取到文本，可能是扫描件且 OCR 未启用")
    return ParsedDocument(
        title=title or path.stem,
        source_type="pdf",
        blocks=blocks,
        page_count=page_count,
        metadata={
            "parser": "pymupdf",
            "pdf_metadata": metadata if pymupdf else {},
            "images": image_count,
            "tables": table_count,
            "chars": sum(len(b.text) for b in blocks),
        },
        warnings=warnings,
    )


def _parse_image(
    path: Path, name: str, settings: Settings, vision: VisionService
) -> ParsedDocument:
    result = vision.extract(path)
    text = result.combined_text() or f"【图片】{path.stem}"
    blocks = [
        Block(
            text=text,
            content_type="vlm" if result.provider.startswith("vlm") else "image",
            metadata={
                "provider": result.provider,
                "caption": result.caption,
                "confidence": result.confidence,
            },
        )
    ]
    return ParsedDocument(
        title=path.stem,
        source_type="image",
        blocks=blocks,
        page_count=0,
        metadata={
            "parser": "image",
            "provider": result.provider,
            "caption": result.caption,
            "keywords": result.keywords,
            "chars": len(text),
        },
        warnings=result.warnings,
    )


def _parse_text(path: Path, name: str, source_type: str = "text") -> ParsedDocument:
    text = _read_text(path)
    blocks: list[Block] = []
    for part in re.split(r"\n\s*\n", text):
        body = clean_extracted_text(part)
        if body:
            blocks.append(Block(text=body, content_type="text"))
    return ParsedDocument(
        title=path.stem,
        source_type=source_type,
        blocks=blocks,
        metadata={"parser": source_type, "chars": len(text)},
    )


def _parse_docx(
    path: Path, name: str, settings: Settings, vision: VisionService, doc_id: str
) -> ParsedDocument:
    blocks: list[Block] = []
    warnings: list[str] = []
    stack = HeadingStack()
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            body = root.find("w:body", namespace)
            if body is not None:
                for child in body:
                    tag = child.tag.rsplit("}", 1)[-1]
                    if tag == "p":
                        texts = [node.text or "" for node in child.findall(".//w:t", namespace)]
                        text = normalize_text("".join(texts))
                        if not text:
                            continue
                        style = ""
                        style_node = child.find("./w:pPr/w:pStyle", namespace)
                        if style_node is not None:
                            style = style_node.get(
                                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", ""
                            )
                        heading_match = re.search(r"heading\s*(\d)|标题\s*(\d)", style, re.I)
                        if heading_match:
                            level = int(heading_match.group(1) or heading_match.group(2) or 1)
                            stack.update(level, text)
                        else:
                            blocks.append(
                                Block(text=text, heading_path=stack.current, content_type="text")
                            )
                    elif tag == "tbl":
                        rows: list[str] = []
                        for row in child.findall(".//w:tr", namespace):
                            cells = []
                            for cell in row.findall("./w:tc", namespace):
                                cell_text = "".join(
                                    node.text or "" for node in cell.findall(".//w:t", namespace)
                                )
                                cells.append(normalize_text(cell_text))
                            if any(cells):
                                rows.append(" | ".join(cells))
                        if rows:
                            blocks.append(
                                Block(text="\n".join(rows), heading_path=stack.current, content_type="table")
                            )
            image_entries = [
                entry for entry in archive.namelist() if entry.startswith("word/media/")
            ][: settings.vlm_max_images_per_doc]
            if image_entries:
                out_dir = settings.extracted_dir / doc_id
                out_dir.mkdir(parents=True, exist_ok=True)
                for entry in image_entries:
                    suffix = Path(entry).suffix.lower()
                    if suffix not in _IMAGE_SUFFIXES:
                        continue
                    target = out_dir / f"docx_{safe_filename(Path(entry).name)}"
                    target.write_bytes(archive.read(entry))
                    result = vision.extract(target)
                    if result.combined_text():
                        blocks.append(
                            Block(
                                text=result.combined_text(),
                                heading_path=stack.current,
                                content_type="vlm" if result.provider.startswith("vlm") else "image",
                                metadata={"provider": result.provider, "source": entry},
                            )
                        )
                        warnings.extend(result.warnings)
    except Exception as exc:
        raise ValueError(f"DOCX 解析失败: {exc}") from exc

    title = path.stem
    if blocks:
        first = blocks[0].text.splitlines()[0]
        if 3 <= len(first) <= 60:
            title = first
    return ParsedDocument(
        title=title,
        source_type="docx",
        blocks=blocks,
        metadata={"parser": "docx", "chars": sum(len(b.text) for b in blocks)},
        warnings=warnings,
    )


def load_document(
    path: Path,
    *,
    name: str,
    settings: Settings,
    vision: VisionService,
    doc_id: str,
) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path, name, settings, vision, doc_id)
    if suffix in {".md", ".markdown"}:
        return _parse_markdown(path, name, settings, vision)
    if suffix in _IMAGE_SUFFIXES:
        return _parse_image(path, name, settings, vision)
    if suffix == ".docx":
        return _parse_docx(path, name, settings, vision, doc_id)
    if suffix in {".txt", ".text"}:
        return _parse_text(path, name, "text")
    if suffix == ".csv":
        return _parse_text(path, name, "csv")
    if suffix == ".json":
        return _parse_text(path, name, "json")
    raise ValueError(f"暂不支持的文件类型: {suffix or 'unknown'}")
