"""Image understanding: local OCR + optional VLM + sidecar fallback.

Provider order
--------------
1. RapidOCR (bundled ONNX model, fully offline, strong Chinese support)
2. Tesseract (if a binary is installed)
3. VLM through an OpenAI-compatible vision model
4. A sidecar ``*.txt`` / ``*.md`` caption next to the image
5. Filename-only placeholder (still indexed and retrievable)
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.core.utils import parse_json_object, truncate
from app.llm.base import BaseLLM
from app.llm.prompts import VISION_SYSTEM

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    caption: str = ""
    text: str = ""
    keywords: list[str] = field(default_factory=list)
    provider: str = "none"
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def combined_text(self) -> str:
        parts: list[str] = []
        if self.caption:
            parts.append(f"【图像描述】{self.caption}")
        if self.text:
            parts.append(f"【图中文字】{self.text}")
        if self.keywords:
            parts.append("【关键词】" + "、".join(self.keywords))
        return "\n".join(parts).strip()


class VisionService:
    _rapid_engine = None
    _rapid_lock = threading.Lock()
    _rapid_failed = False

    def __init__(self, settings, llm: BaseLLM):
        self.settings = settings
        self.llm = llm

    # ------------------------------------------------------------------ OCR
    def _rapidocr(self, path: Path) -> tuple[str, str] | None:
        if not self.settings.ocr_enabled or self._rapid_failed:
            return None
        with self._rapid_lock:
            if self.__class__._rapid_engine is None:
                try:
                    try:
                        from rapidocr_onnxruntime import RapidOCR  # type: ignore
                    except Exception:
                        from rapidocr import RapidOCR  # type: ignore
                    self.__class__._rapid_engine = RapidOCR()
                except Exception as exc:
                    self.__class__._rapid_failed = True
                    logger.info("RapidOCR 不可用: %s", exc)
                    return None
            engine = self.__class__._rapid_engine
        try:
            result, _elapsed = engine(str(path))
            if not result:
                return None
            lines: list[str] = []
            for item in result:
                # rapidocr returns [box, text, score]
                text = str(item[1]).strip() if len(item) > 1 else ""
                score = float(item[2]) if len(item) > 2 else 1.0
                if text and score >= 0.45:
                    lines.append(text)
            if not lines:
                return None
            return "\n".join(lines), "rapidocr"
        except Exception as exc:
            logger.info("RapidOCR 识别失败: %s", exc)
            return None

    def _tesseract(self, path: Path) -> tuple[str, str] | None:
        if not self.settings.ocr_enabled:
            return None
        try:
            import pytesseract  # type: ignore

            if self.settings.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd
            with Image.open(path) as image:
                text = pytesseract.image_to_string(image, lang=self.settings.ocr_lang)
            text = text.strip()
            return (text, "tesseract") if text else None
        except Exception as exc:
            logger.debug("Tesseract 不可用: %s", exc)
            return None

    def _ocr(self, path: Path) -> tuple[str, str] | None:
        return self._rapidocr(path) or self._tesseract(path)

    # ------------------------------------------------------------------ VLM
    def _image_data_url(self, path: Path) -> str:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            max_side = 1680
            if max(width, height) > max_side:
                scale = max_side / float(max(width, height))
                image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=86)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _vlm_result(text: str) -> VisionResult:
        data = parse_json_object(text) or {}
        return VisionResult(
            caption=truncate(str(data.get("caption") or "").strip(), 300),
            text=str(data.get("text") or "").strip()[:3000],
            keywords=[str(k) for k in (data.get("keywords") or [])][:10],
            provider="vlm",
            confidence=0.82,
        )

    def _vlm(self, path: Path) -> VisionResult | None:
        if not (self.settings.enable_vlm and self.llm.supports_vision):
            return None
        try:
            data_url = self._image_data_url(path)
            messages = [
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请解析这张文档图片。只输出 JSON。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ]
            result = self.llm.chat(
                messages,
                json_mode=True,
                max_tokens=900,
                model=self.settings.llm_vision_model or None,
            )
            return self._vlm_result(result.text)
        except Exception as exc:
            # Some OpenAI-compatible gateways reject response_format for
            # vision models.  Retry once as plain text and parse JSON from the
            # answer; auth/rate-limit errors should still fail fast.
            text = str(exc).lower()
            if any(
                token in text
                for token in ("400", "response_format", "json_object", "json mode")
            ):
                try:
                    logger.info("VLM JSON 模式不可用，改用文本模式重试: %s", exc)
                    result = self.llm.chat(
                        messages,
                        json_mode=False,
                        max_tokens=900,
                        model=self.settings.llm_vision_model or None,
                    )
                    return self._vlm_result(result.text)
                except Exception as retry_exc:
                    logger.info("VLM 文本模式重试失败，回退 OCR: %s", retry_exc)
                    return None
            logger.info("VLM 解析失败，回退 OCR: %s", exc)
            return None

    # -------------------------------------------------------------- sidecar
    @staticmethod
    def _sidecar(path: Path) -> str:
        candidates = [
            path.with_suffix(path.suffix + ".txt"),
            path.with_suffix(".md"),
            path.parent / f"{path.stem}.caption.txt",
            path.parent / f"{path.stem}.caption.md",
        ]
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
        return ""

    # --------------------------------------------------------------- extract
    def extract(self, path: Path, *, allow_vlm: bool = True) -> VisionResult:
        warnings: list[str] = []
        ocr_text = ""
        ocr_provider = ""
        ocr_result = self._ocr(path)
        if ocr_result:
            ocr_text, ocr_provider = ocr_result

        vlm_result = self._vlm(path) if allow_vlm else None
        if vlm_result and vlm_result.text:
            text = vlm_result.text
            if ocr_text and ocr_text not in text:
                # VLM text is usually cleaner; append OCR only if materially different.
                if len(ocr_text) > len(text) * 1.4:
                    text = ocr_text
            provider = f"vlm+{ocr_provider}" if ocr_text else "vlm"
            caption = vlm_result.caption
            keywords = vlm_result.keywords
            confidence = vlm_result.confidence + (0.05 if ocr_text else 0.0)
        elif ocr_text:
            text = ocr_text
            provider = ocr_provider
            caption = ""
            keywords = []
            confidence = 0.7
        else:
            sidecar = self._sidecar(path)
            if sidecar:
                text = sidecar
                caption = ""
                provider = "sidecar"
                keywords = []
                confidence = 0.55
                warnings.append("未提取到图片文字，使用图片旁的说明文件")
            else:
                text = ""
                caption = f"图片文件：{path.stem}"
                provider = "filename"
                keywords = []
                confidence = 0.12
                warnings.append("未提取到图片文字：请安装 RapidOCR、配置 VLM 或添加 sidecar")

        return VisionResult(
            caption=caption,
            text=text,
            keywords=keywords,
            provider=provider,
            confidence=min(0.95, confidence),
            warnings=warnings,
        )

    def status(self) -> dict:
        rapid = "ready" if self._rapid_engine is not None else ("unavailable" if self._rapid_failed else "lazy")
        return {
            "ocr_enabled": self.settings.ocr_enabled,
            "rapidocr": rapid,
            "tesseract_cmd": self.settings.tesseract_cmd or "auto",
            "vlm_enabled": bool(self.settings.enable_vlm and self.llm.supports_vision),
            "vlm_model": self.settings.llm_vision_model or "",
            "vlm_base_url": getattr(self.llm, "base_url", ""),
        }
