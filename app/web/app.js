/* ============================================================
   知屿 web console
   Vanilla JS, no build step. Streams SSE from FastAPI.
   ============================================================ */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

// Screenshot / reduced-motion mode: `?still=1` disables all entrance animations.
const initialParams = new URLSearchParams(location.search);
if (initialParams.get('still') === '1') {
  document.documentElement.classList.add('still');
}
if (initialParams.get('token')) {
  localStorage.setItem('zhiyu.token', initialParams.get('token'));
}

const THEME_STORAGE_KEY = 'zhiyu.theme';
const THEMES = [
  {
    id: 'default',
    name: '默认 · 白绿渐变',
    desc: '白色基底与青绿渐层，安静克制，适合长时间阅读。',
  },
  {
    id: 'alice',
    name: '艾莉丝',
    desc: '雾蓝与浅紫的柔和渐变，像一本带插画的夜读笔记。',
  },
  {
    id: 'big-fish',
    name: '大肥鱼',
    desc: '深海女仆工坊：深海蓝、陶瓷白与柔金，鲸鱼娘的蕾丝与玻璃质感。',
  },
];

function readThemeId() {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return THEMES.some((theme) => theme.id === saved) ? saved : 'default';
  } catch (error) {
    return 'default';
  }
}

function authHeaders() {
  const token = localStorage.getItem('zhiyu.token');
  return token ? { 'X-API-Key': token } : {};
}

const ICONS = {
  thumbUp: '<svg viewBox="0 0 24 24"><path d="M7 10v10H4V10zM7 10l4-7c1.4 0 2.3 1 2 2.4L12 9h6.2c1.2 0 2 1.1 1.7 2.2l-1.6 6.2A2.5 2.5 0 0 1 15.9 19H7"/></svg>',
  thumbDown: '<svg viewBox="0 0 24 24"><path d="M17 14V4h3v10zM17 14l-4 7c-1.4 0-2.3-1-2-2.4L12 15H5.8c-1.2 0-2-1.1-1.7-2.2l1.6-6.2A2.5 2.5 0 0 1 8.1 5H17"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v6M14 11v6"/></svg>',
  arrow: '<svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
};

const state = {
  view: 'chat',
  sessionId: localStorage.getItem('zhiyu.session') || '',
  sessions: [],
  docs: [],
  system: null,
  suggestions: [],
  streaming: false,
  currentMessageId: '',
  currentSources: [],
  currentTrace: [],
  activeAssistant: null,
  currentQuestion: '',
  traceOpen: false,
  evalRun: null,
  evalTimer: null,
  pollTimer: null,
  systemTab: 'intro',
};

const TITLES = {
  chat: ['WORKSPACE', '对话'],
  library: ['KNOWLEDGE BASE', '知识库'],
  eval: ['EVALUATION', '评测'],
  system: ['SYSTEM', '系统'],
};

const INTENT_LABELS = {
  kb_question: '知识问答',
  chitchat: '寒暄',
  meta: '知识库工具',
  summarize: '总结',
  compare: '对比',
};

/* ------------------------------------------------------------------ utils */

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatTime(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString('zh-CN', {
      month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return '';
  }
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits).replace(/\.?0+$/, '') || '0';
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(2);
}

function formatMs(value) {
  if (!value && value !== 0) return '—';
  return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`;
}

function cleanSnippet(text, limit = 150) {
  const value = String(text || '').replace(/\s+/g, ' ').trim();
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function toast(message, type = '') {
  const el = $('#toast');
  el.textContent = message;
  el.className = `toast show ${type}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    el.className = 'toast';
  }, 3200);
}

function confirmDialog(title, text, confirmLabel = '确认') {
  return new Promise((resolve) => {
    const backdrop = $('#modal-backdrop');
    $('#modal-title').textContent = title;
    $('#modal-text').textContent = text;
    $('#modal-confirm').textContent = confirmLabel;
    backdrop.hidden = false;
    const cleanup = () => {
      backdrop.hidden = true;
      $('#modal-confirm').removeEventListener('click', onConfirm);
      $('#modal-cancel').removeEventListener('click', onCancel);
    };
    const onConfirm = () => { cleanup(); resolve(true); };
    const onCancel = () => { cleanup(); resolve(false); };
    $('#modal-confirm').addEventListener('click', onConfirm);
    $('#modal-cancel').addEventListener('click', onCancel);
  });
}

async function api(path, options = {}) {
  const opts = {
    method: options.method || 'GET',
    headers: { ...authHeaders(), ...(options.headers || {}) },
  };
  if (options.body instanceof FormData) {
    opts.body = options.body;
  } else if (options.body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, opts);
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const message = (data && (data.detail || data.message)) || `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return data;
}

/* ---------------------------------------------------------------- markdown */

function mdInline(text) {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d{1,2})\]/g, '<button class="cite" data-cite="$1" title="查看来源 $1">$1</button>');
}

function renderMarkdown(text) {
  const source = String(text || '');
  const codeBlocks = [];
  const withoutCode = source.replace(/```[a-zA-Z0-9_-]*\n?([\s\S]*?)```/g, (_, code) => {
    codeBlocks.push(code.replace(/\n$/, ''));
    return `\u0000CODE${codeBlocks.length - 1}\u0000`;
  });

  const lines = withoutCode.split('\n');
  let html = '';
  let paragraph = [];
  let listType = '';
  let listItems = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html += `<p>${mdInline(paragraph.map(escapeHtml).join('<br>'))}</p>`;
    paragraph = [];
  };
  const flushList = () => {
    if (!listItems.length) return;
    html += `<${listType}>${listItems.map((item) => `<li>${mdInline(escapeHtml(item))}</li>`).join('')}</${listType}>`;
    listType = '';
    listItems = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = trimmed.match(/^#{1,4}\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      html += `<h3>${mdInline(escapeHtml(heading[1]))}</h3>`;
      continue;
    }
    const bullet = trimmed.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      if (listType && listType !== 'ul') flushList();
      listType = 'ul';
      listItems.push(bullet[1]);
      continue;
    }
    const numbered = trimmed.match(/^(\d+)[.、]\s+(.*)$/);
    if (numbered) {
      flushParagraph();
      if (listType && listType !== 'ol') flushList();
      listType = 'ol';
      listItems.push(numbered[2]);
      continue;
    }
    if (trimmed.startsWith('>')) {
      flushParagraph();
      flushList();
      html += `<blockquote>${mdInline(escapeHtml(trimmed.replace(/^>\s?/, '')))}</blockquote>`;
      continue;
    }
    if (listType && (trimmed.startsWith('|') || /^\S/.test(trimmed)) && paragraph.length === 0) {
      // A plain line after a list closes it.
      flushList();
    }
    paragraph.push(line);
  }
  flushParagraph();
  flushList();

  html = html.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => {
    return `<pre><code>${escapeHtml(codeBlocks[Number(index)] || '')}</code></pre>`;
  });
  return html || '<p></p>';
}

/* --------------------------------------------------------------- data load */

async function loadSystem() {
  try {
    const data = await api('/api/v1/system/info');
    state.system = data;
    renderGlobalStatus(data);
    renderSystemView(data);
  } catch (error) {
    $('#provider-name').textContent = '连接失败';
    $('#provider-sub').textContent = error.message;
    $('#status-dot').classList.add('warn');
  }
}

function renderGlobalStatus(data) {
  const { app, llm, index, stats, cache } = data;
  $('#provider-name').textContent = llm.mock_mode ? '本地离线模式' : (llm.model || llm.provider);
  $('#provider-sub').textContent = llm.mock_mode ? '抽取式回答 · 无需 API Key' : `${llm.provider} · ${llm.model}`;
  $('#status-dot').classList.toggle('warn', Boolean(llm.mock_mode));
  $('#index-version').textContent = index.index_version;
  $('#cache-pill').textContent = `缓存 ${cache.exact_entries + cache.semantic_entries}`;
  $('#hero-docs').textContent = stats.documents;
  $('#hero-chunks').textContent = stats.chunks;
  $('#hero-version').textContent = `v${index.index_version}`;
  $('#lib-docs').textContent = stats.documents;
  $('#lib-chunks').textContent = stats.chunks;
  $('#lib-version').textContent = index.index_version;
  $('#lib-vector').textContent = index.vectors.backend;
  $('#lib-embedding').textContent = index.embedder.name;
}

async function loadDocuments() {
  try {
    const data = await api('/api/v1/documents');
    state.docs = data.documents || [];
    renderDocumentTable();
    renderDocumentFilter();
    if (data.stats) {
      $('#hero-docs').textContent = data.stats.documents;
      $('#hero-chunks').textContent = data.stats.chunks;
      $('#lib-docs').textContent = data.stats.documents;
      $('#lib-chunks').textContent = data.stats.chunks;
      $('#lib-version').textContent = data.index_version;
    }
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function loadSuggestions() {
  try {
    const data = await api('/api/v1/suggestions');
    state.suggestions = data.suggestions || [];
    renderSuggestions();
  } catch {
    state.suggestions = [
      { text: '混合检索是怎么做融合的？' },
      { text: '扫描版 PDF 是如何解析的？' },
      { text: '知识库里有哪些文档？' },
    ];
    renderSuggestions();
  }
}

async function loadSessions() {
  try {
    const data = await api('/api/v1/sessions?client=web&limit=30');
    state.sessions = data.sessions || [];
    renderSessions();
  } catch {
    /* sidebar history is non-critical */
  }
}

/* ------------------------------------------------------------------- hero */

function renderSuggestions() {
  const wrap = $('#suggestions');
  wrap.innerHTML = '';
  state.suggestions.slice(0, 6).forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'suggest-chip';
    button.textContent = item.text;
    button.addEventListener('click', () => {
      $('#message-input').value = item.text;
      autoResize();
      sendCurrent();
    });
    wrap.appendChild(button);
  });
}

function renderSessions() {
  const wrap = $('#session-list');
  wrap.innerHTML = '';
  if (!state.sessions.length) {
    wrap.innerHTML = '<div class="session-empty">还没有对话记录</div>';
    return;
  }
  state.sessions.forEach((session) => {
    const button = document.createElement('div');
    button.className = `session-item${session.id === state.sessionId ? ' active' : ''}`;
    button.setAttribute('role', 'button');
    button.tabIndex = 0;
    button.innerHTML = `<strong>${escapeHtml(session.title || '未命名对话')}</strong><span>${formatTime(session.updated_at)} · ${session.message_count || 0} 条</span>`;
    button.addEventListener('click', () => openSession(session.id));
    button.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') openSession(session.id);
    });
    const remove = document.createElement('button');
    remove.className = 'session-del';
    remove.title = '删除对话';
    remove.textContent = '×';
    remove.addEventListener('click', async (event) => {
      event.stopPropagation();
      const confirmed = await confirmDialog('删除对话', '这条对话记录及其消息会从本地删除，无法恢复。', '删除');
      if (!confirmed) return;
      try {
        await api(`/api/v1/sessions/${session.id}`, { method: 'DELETE' });
        if (state.sessionId === session.id) newChat();
        await loadSessions();
        toast('对话已删除');
      } catch (error) {
        toast(error.message, 'error');
      }
    });
    button.appendChild(remove);
    wrap.appendChild(button);

  });
}

/* --------------------------------------------------------------- messages */

function hideHero() {
  const hero = $('#hero');
  if (hero) hero.hidden = true;
  $('#suggestions').classList.add('hidden');
}

function showHero() {
  const hero = $('#hero');
  if (hero) hero.hidden = false;
  $('#suggestions').classList.remove('hidden');
}

function clearChat() {
  $$('.message', $('#chat-stream')).forEach((el) => el.remove());
  showHero();
  scrollChatToBottom(true);
}

function scrollChatToBottom(force = false) {
  const stream = $('#chat-stream');
  if (!stream) return;
  const threshold = 180;
  const nearBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight < threshold;
  if (force || nearBottom) {
    stream.scrollTo({ top: stream.scrollHeight, behavior: force ? 'auto' : 'smooth' });
  }
}

function createUserMessage(text) {
  const node = document.createElement('div');
  node.className = 'message user';
  node.innerHTML = `<div class="avatar">你</div><div class="message-body"><div class="user-bubble">${escapeHtml(text)}</div></div>`;
  $('#chat-stream').appendChild(node);
  return node;
}

function createAssistantShell(mode = 'stream') {
  const node = document.createElement('div');
  node.className = 'message assistant';
  node.innerHTML = `
    <div class="avatar">知</div>
    <div class="message-body">
      <div class="assistant-name"><strong>知屿</strong><span>RAG AGENT</span></div>
      <div class="answer-body">${mode === 'stream' ? '<div class="typing"><i></i><i></i><i></i></div>' : ''}</div>
      <div class="citation-list"></div>
      <div class="answer-meta"></div>
    </div>`;
  $('#chat-stream').appendChild(node);
  return node;
}

function confidenceClass(value, label) {
  if (label === '高' || value >= 0.7) return 'high';
  if (label === '中' || value >= 0.5) return 'mid';
  return 'low';
}

function renderMeta(node, data) {
  const meta = $('.answer-meta', node);
  if (!meta) return;
  const value = Number(data.confidence || 0);
  const label = data.confidence_label || (value >= 0.7 ? '高' : value >= 0.5 ? '中' : '低');
  const cls = confidenceClass(value, label);
  const bits = [];
  bits.push(`<span class="confidence ${cls}">${label}置信度 · ${formatNumber(value)}</span>`);
  if (data.latency_ms !== undefined) bits.push(`<span class="meta-text">${formatMs(data.latency_ms)}</span>`);
  if (data.citation_report && data.citation_report.total) {
    bits.push(`<span class="meta-text">引用 ${data.citation_report.supported}/${data.citation_report.total}</span>`);
  }
  if (data.retry_used) bits.push('<span class="meta-text">二次检索</span>');
  if (data.cached) bits.push('<span class="meta-text">缓存命中</span>');
  bits.push(`<button class="meta-link" data-trace-action="open">查看轨迹</button>`);
  bits.push(`<div class="feedback">
      <button title="有帮助" data-feedback="1">${ICONS.thumbUp}</button>
      <button title="没帮助" data-feedback="-1">${ICONS.thumbDown}</button>
    </div>`);
  meta.innerHTML = bits.join('');
  const messageId = node.dataset.messageId || '';
  meta.querySelectorAll('[data-feedback]').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!messageId) return;
      try {
        await api(`/api/v1/messages/${messageId}/feedback`, {
          method: 'POST',
          body: { message_id: messageId, rating: Number(button.dataset.feedback), comment: '' },
        });
        meta.querySelectorAll('[data-feedback]').forEach((el) => el.classList.remove('active'));
        button.classList.add('active');
        toast('感谢反馈');
      } catch (error) {
        toast(error.message, 'error');
      }
    });
  });
  meta.querySelector('[data-trace-action]')?.addEventListener('click', () => {
    openTrace();
    if (state.currentTrace.length) renderTrace(state.currentTrace);
  });
}

function renderCitations(node, sources, answerText = '') {
  const wrap = $('.citation-list', node);
  if (!wrap) return;
  wrap.innerHTML = '';
  if (!sources || !sources.length) return;
  const cited = new Set();
  for (const match of String(answerText || '').matchAll(/\[(\d{1,2})\]/g)) {
    cited.add(Number(match[1]));
  }
  const visible = cited.size
    ? sources.filter((source) => cited.has(Number(source.index)))
    : sources;
  (visible.length ? visible : sources).forEach((source) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'citation-card';
    item.dataset.citeIndex = source.index;
    item.innerHTML = `
      <span class="citation-index">${source.index}</span>
      <span class="citation-copy">
        <strong>${escapeHtml(source.doc_name || '未知文档')}</strong>
        <span>${escapeHtml(source.heading || '正文')}${source.page_start ? ` · 第 ${source.page_start} 页` : ''}</span>
      </span>
      <span class="citation-score">${formatNumber(source.score || source.rerank_score || 0)}</span>`;
    item.addEventListener('click', () => {
      openTrace();
      highlightSource(source.index);
    });
    wrap.appendChild(item);
  });
}

function renderAssistantResult(node, data) {
  const body = $('.answer-body', node);
  body.classList.remove('streaming');
  body.innerHTML = renderMarkdown(data.answer || '');
  body.querySelectorAll('[data-cite]').forEach((button) => {
    button.addEventListener('click', () => {
      openTrace();
      highlightSource(Number(button.dataset.cite));
    });
  });
  node.dataset.messageId = data.message_id || node.dataset.messageId || '';
  node.dataset.trace = JSON.stringify(data.trace || []);
  state.currentSources = data.sources || data.citations || [];
  renderCitations(node, state.currentSources);
  renderMeta(node, data);
  if (data.trace && data.trace.length) {
    state.currentTrace = data.trace;
    renderTrace(data.trace);
  }
  if (state.currentSources.length) renderTraceSummary(data);
  if (state.traceAutoOpen) openTrace();
}

function appendHistoryMessage(message) {
  if (message.role === 'user') {
    createUserMessage(message.content);
    return;
  }
  const node = createAssistantShell('static');
  node.dataset.messageId = message.id || '';
  node.dataset.trace = JSON.stringify(message.trace || []);
  const body = $('.answer-body', node);
  body.innerHTML = renderMarkdown(message.content || '');
  body.querySelectorAll('[data-cite]').forEach((button) => {
    button.addEventListener('click', () => {
      openTrace();
      highlightSource(Number(button.dataset.cite));
    });
  });
  const citations = message.citations || [];
  renderCitations(node, citations, message.content || '');
  renderMeta(node, {
    confidence: message.confidence,
    confidence_label: message.confidence >= 0.7 ? '高' : message.confidence >= 0.5 ? '中' : '低',
    latency_ms: message.latency_ms,
    cached: message.cached,
    citation_report: null,
  });
  $('#chat-stream').appendChild(node);
}

async function openSession(sessionId) {
  if (state.streaming) return;
  try {
    const data = await api(`/api/v1/sessions/${sessionId}/messages?limit=120`);
    state.sessionId = sessionId;
    localStorage.setItem('zhiyu.session', sessionId);
    $$('.message', $('#chat-stream')).forEach((el) => el.remove());
    state.currentTrace = [];
    resetTrace();
    (data.messages || []).forEach(appendHistoryMessage);
    const hasMessages = Boolean(data.messages && data.messages.length);
    if (hasMessages) {
      hideHero();
      $('#suggestions').classList.add('hidden');
    } else {
      showHero();
    }
    const lastAssistant = [...(data.messages || [])].reverse().find((m) => m.role === 'assistant');
    state.currentSources = lastAssistant?.citations || [];
    state.currentConfidence = lastAssistant?.confidence ?? 0;
    state.currentConfidenceLabel = (lastAssistant?.confidence ?? 0) >= 0.7 ? '高' : (lastAssistant?.confidence ?? 0) >= 0.5 ? '中' : '低';
    state.currentReport = null;
    if (state.currentSources.length) {
      renderTraceSources(state.currentSources);
      renderTraceSummary({
        confidence: state.currentConfidence,
        confidence_label: state.currentConfidenceLabel,
        citation_report: null,
        latency_ms: lastAssistant?.latency_ms,
        plan: null,
      });
    }
    if (lastAssistant && lastAssistant.trace && lastAssistant.trace.length) {
      state.currentTrace = lastAssistant.trace;
      renderTrace(lastAssistant.trace, true);
    }
    renderSessions();
    scrollChatToBottom(true);
    if (window.innerWidth <= 1024) $('#rail').classList.remove('open');
  } catch (error) {
    toast(error.message, 'error');
  }
}

function newChat() {
  state.sessionId = '';
  state.currentMessageId = '';
  state.currentSources = [];
  state.currentTrace = [];
  localStorage.removeItem('zhiyu.session');
  clearChat();
  resetTrace();
  renderSessions();
  $('#message-input').focus();
}

/* ------------------------------------------------------------------- chat */

function autoResize() {
  const input = $('#message-input');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

function selectedDocIds() {
  const value = $('#doc-filter').value;
  return value ? [value] : undefined;
}

function sendCurrent() {
  const input = $('#message-input');
  const text = input.value.trim();
  if (!text || state.streaming) return;
  input.value = '';
  autoResize();
  sendMessage(text);
}

async function sendMessage(text) {
  state.streaming = true;
  state.currentQuestion = text;
  state.currentSources = [];
  state.currentTrace = [];
  state.currentMessageId = '';
  $('#send-btn').disabled = true;
  hideHero();
  createUserMessage(text);
  const assistant = createAssistantShell('stream');
  state.activeAssistant = assistant;
  resetTrace();
  scrollChatToBottom(true);

  try {
    await streamChat(text, assistant);
  } catch (error) {
    const body = $('.answer-body', assistant);
    body.classList.remove('streaming');
    body.innerHTML = `<div class="error-card">回答失败：${escapeHtml(error.message)}</div>`;
    toast(error.message, 'error');
  } finally {
    state.streaming = false;
    $('#send-btn').disabled = false;
    scrollChatToBottom();
    loadSessions();
    loadSystem();
  }
}

async function streamChat(text, assistant) {
  const body = $('.answer-body', assistant);
  const payload = {
    message: text,
    session_id: state.sessionId || null,
    doc_ids: selectedDocIds() || null,
  };
  const response = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    let message = `请求失败 (${response.status})`;
    try {
      const data = await response.json();
      message = data.detail || data.message || message;
    } catch { /* ignore */ }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let answer = '';
  let gotToken = false;
  let finished = false;

  const consume = (block) => {
    const lines = block.split('\n');
    let event = 'message';
    let dataText = '';
    for (const line of lines) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataText += line.slice(5).trim();
    }
    if (!dataText) return;
    let data;
    try { data = JSON.parse(dataText); } catch { return; }

    if (event === 'meta') {
      state.sessionId = data.session_id || state.sessionId;
      if (state.sessionId) localStorage.setItem('zhiyu.session', state.sessionId);
      state.currentMessageId = data.message_id || '';
      assistant.dataset.messageId = data.message_id || '';
    } else if (event === 'trace') {
      state.currentTrace.push(data);
      appendTraceStep(data);
    } else if (event === 'token') {
      gotToken = true;
      answer += data.text || '';
      body.classList.add('streaming');
      body.textContent = answer;
      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      body.appendChild(cursor);
      scrollChatToBottom();
    } else if (event === 'sources') {
      state.currentSources = data.sources || data.citations || [];
      state.currentConfidence = data.confidence;
      state.currentConfidenceLabel = data.confidence_label;
      state.currentReport = data.citation_report;
      renderCitations(assistant, state.currentSources, answer);
      renderMeta(assistant, {
        ...data,
        message_id: state.currentMessageId,
      });
      renderTraceSources(state.currentSources);
      renderTraceSummary(data);
      if (state.traceAutoOpen) openTrace();
      if (answer) {
        body.classList.remove('streaming');
        body.innerHTML = renderMarkdown(answer);
        body.querySelectorAll('[data-cite]').forEach((button) => {
          button.addEventListener('click', () => {
            openTrace();
            highlightSource(Number(button.dataset.cite));
          });
        });
      }
    } else if (event === 'done') {
      finished = true;
      assistant.dataset.messageId = data.message_id || state.currentMessageId;
      if (state.currentSources.length) {
        body.innerHTML = renderMarkdown(answer);
        body.querySelectorAll('[data-cite]').forEach((button) => {
          button.addEventListener('click', () => {
            openTrace();
            highlightSource(Number(button.dataset.cite));
          });
        });
      }
      renderMeta(assistant, {
        ...data,
        confidence: state.currentConfidence ?? 0,
        confidence_label: state.currentConfidenceLabel || '',
        citation_report: state.currentReport || null,
      });
    } else if (event === 'error') {
      finished = true;
      body.classList.remove('streaming');
      body.innerHTML = `<div class="error-card">${escapeHtml(data.message || '生成失败')}</div>`;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      consume(block);
    }
  }
  if (buffer.trim()) consume(buffer);
  if (!gotToken && !finished) {
    body.classList.remove('streaming');
    body.innerHTML = '<div class="error-card">没有收到回答内容，请重试。</div>';
  }
}

/* ------------------------------------------------------------- trace panel */

function resetTrace() {
  $('#trace-timeline').innerHTML = '';
  $('#trace-sources').innerHTML = '';
  $('#trace-summary').innerHTML = `
    <div class="trace-empty">
      <div class="trace-orbit"><span></span><span></span><span></span></div>
      <p>发送一个问题，这里会展示查询改写、多路召回、重排、生成与引用核查的完整链路。</p>
    </div>`;
}

function openTrace() {
  $('#view-chat').classList.add('trace-open');
  state.traceOpen = true;
}

function toggleTrace() {
  $('#view-chat').classList.toggle('trace-open');
  state.traceOpen = $('#view-chat').classList.contains('trace-open');
}

function detailText(step) {
  const detail = step.detail || {};
  const name = step.name;
  if (name === 'analyze') {
    const queries = (detail.subqueries || []).join(' / ');
    return `意图 <b>${INTENT_LABELS[detail.intent] || detail.intent}</b> · 改写「${escapeHtml(detail.rewritten || '')}」${queries ? `<br>子查询 ${escapeHtml(queries)}` : ''}`;
  }
  if (name === 'retrieve') {
    return `查询 ${detail.queries ? detail.queries.length : 0} 路 · 向量 ${detail.dense_hits || 0} · BM25 ${detail.bm25_hits || 0} · 候选 ${detail.fused_candidates || 0} · 相关性 ${formatNumber(detail.relevance || 0)}`;
  }
  if (name === 'context') {
    return `选用 ${detail.chunks || 0} 个片段 · 约 ${detail.tokens || 0} tokens${detail.expanded ? ` · 补全 ${detail.expanded} 个相邻片段` : ''}`;
  }
  if (name === 'retry') {
    return `${detail.before ?? '—'} → ${detail.after ?? '—'} · 合并 ${detail.merged_chunks || 0} 个候选`;
  }
  if (name === 'second_pass') {
    return `采用：${detail.adopted ? '第二次结果' : '保留第一次结果'} · ${formatNumber(detail.before || 0)} → ${formatNumber(detail.after || 0)}`;
  }
  if (name === 'generate') {
    return `${escapeHtml(detail.provider || 'model')} · ${detail.chars || 0} 字${detail.stream ? ' · SSE' : ''}`;
  }
  if (name === 'citation') {
    const total = detail.total || 0;
    const supported = detail.supported || 0;
    return `引用 ${supported}/${total} 被支持 · 覆盖 ${formatNumber((detail.coverage || 0) * 100, 0)}%`;
  }
  if (name === 'cache') {
    return detail.kind === 'miss' ? '未命中，继续检索' : `${detail.kind} · ${formatNumber(detail.score || 0)}`;
  }
  if (name === 'route') {
    return detail.intent ? `意图 ${escapeHtml(detail.intent)}` : (detail.relevance !== undefined ? `相关性 ${formatNumber(detail.relevance)}` : '');
  }
  if (name === 'persist') return '写入会话历史';
  const keys = Object.keys(detail).slice(0, 3);
  return keys.map((key) => `${escapeHtml(key)}: ${escapeHtml(String(detail[key]).slice(0, 80))}`).join(' · ');
}

function appendTraceStep(step) {
  const summary = $('#trace-summary');
  if (summary.querySelector('.trace-empty')) summary.innerHTML = '';
  const timeline = $('#trace-timeline');
  const node = document.createElement('div');
  node.className = `trace-step ${step.status || 'ok'}`;
  node.innerHTML = `
    <span class="trace-dot"></span>
    <div class="trace-step-body">
      <div class="trace-step-title"><strong>${escapeHtml(step.label || step.name)}</strong><span>${step.duration_ms ? `${step.duration_ms}ms` : ''}</span></div>
      <div class="trace-step-detail">${detailText(step)}</div>
    </div>`;
  timeline.appendChild(node);
}

function renderTrace(trace, quiet = false) {
  resetTrace();
  (trace || []).forEach(appendTraceStep);
  const lastSources = state.currentSources;
  if (lastSources && lastSources.length) {
    renderTraceSources(lastSources);
    renderTraceSummary({
      confidence: lastSources[0]?.score,
      confidence_label: '',
    });
  }
  if (!quiet) openTrace();
}

function renderTraceSummary(data) {
  const sources = state.currentSources || [];
  const top = sources[0] || {};
  const plan = data.plan || {};
  const conf = Number(data.confidence ?? top.score ?? 0);
  const report = data.citation_report || {};
  const summary = $('#trace-summary');
  summary.innerHTML = `
    <div class="trace-card">
      <div class="trace-card-head">
        <strong>${escapeHtml(INTENT_LABELS[plan.intent] || plan.intent || '知识问答')}</strong>
        <span class="trace-intent">${data.retry_used ? '二次检索' : '单轮'}</span>
      </div>
      <div class="trace-query">${escapeHtml(plan.rewritten || state.currentQuestion || '')}</div>
      <div class="trace-kpis">
        <div><span>置信度</span><strong>${formatNumber(conf)}</strong></div>
        <div><span>引用</span><strong>${report.supported ?? 0}/${report.total ?? 0}</strong></div>
        <div><span>延迟</span><strong>${formatMs(data.latency_ms || 0)}</strong></div>
      </div>
    </div>`;
}

function renderTraceSources(sources) {
  const wrap = $('#trace-sources');
  wrap.innerHTML = '<h4>SOURCES · 按相关性排序</h4>';
  (sources || []).forEach((source) => {
    const card = document.createElement('div');
    card.className = 'trace-source';
    card.dataset.index = source.index;
    const score = Number(source.score || source.rerank_score || 0);
    card.innerHTML = `
      <div class="trace-source-top"><b>${source.index}</b><span>${escapeHtml(source.doc_name || '')}</span></div>
      <p>${escapeHtml(source.heading || '正文')}${source.page_start ? ` · 第 ${source.page_start} 页` : ''}</p>
      <div class="score-bar"><i style="width:${Math.max(4, Math.min(100, score * 100))}%"></i></div>`;
    wrap.appendChild(card);
  });
}

function highlightSource(index) {
  $$('.trace-source').forEach((el) => {
    el.style.borderColor = '';
    el.style.background = '';
  });
  const target = $(`.trace-source[data-index="${index}"]`);
  if (target) {
    target.style.borderColor = 'var(--teal)';
    target.style.background = 'var(--teal-wash)';
    target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

/* --------------------------------------------------------------- documents */

function docType(doc) {
  const name = (doc.name || '').toLowerCase();
  if (name.endsWith('.pdf')) return ['PDF', 'pdf'];
  if (name.endsWith('.md') || name.endsWith('.markdown')) return ['MD', 'md'];
  if (/\.(png|jpe?g|webp|gif)$/.test(name)) return ['IMG', 'img'];
  return ['TXT', 'txt'];
}

function renderDocumentTable() {
  const search = ($('#doc-search').value || '').trim().toLowerCase();
  const statusFilter = $('#doc-status-filter').value;
  const docs = state.docs.filter((doc) => {
    if (search && !doc.name.toLowerCase().includes(search)) return false;
    if (statusFilter && doc.status !== statusFilter) return false;
    return true;
  });
  const tbody = $('#doc-tbody');
  tbody.innerHTML = '';
  $('#doc-empty').hidden = docs.length > 0;
  $('#doc-count-label').textContent = `${state.docs.length} 篇资料 · ${state.docs.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0)} 个片段`;
  docs.forEach((doc) => {
    const [badge, cls] = docType(doc);
    const status = doc.status === 'ready' ? 'ready' : doc.status === 'error' ? 'error' : 'processing';
    const statusText = doc.status === 'ready' ? '就绪' : doc.status === 'error' ? '失败' : '处理中';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><div class="doc-cell"><span class="doc-type ${cls}">${badge}</span><span><strong title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</strong><small>${escapeHtml(doc.metadata?.title || doc.source_key || '')}</small></span></div></td>
      <td>${escapeHtml((doc.mime || '').split('/').pop() || badge)}</td>
      <td>${doc.chunk_count || 0}</td>
      <td>${escapeHtml(doc.size_human || '—')}</td>
      <td><span class="status-pill ${status}">${statusText}</span></td>
      <td>${formatTime(doc.updated_at)}</td>
      <td><div class="row-actions">
        <button class="row-btn" data-action="reindex" title="重建索引">${ICONS.refresh}</button>
        <button class="row-btn danger" data-action="delete" title="删除">${ICONS.trash}</button>
      </div></td>`;
    row.querySelector('[data-action="reindex"]')?.addEventListener('click', () => reindexDocument(doc));
    row.querySelector('[data-action="delete"]')?.addEventListener('click', () => deleteDocument(doc));
    if (doc.status === 'error' && doc.error) row.title = doc.error;
    tbody.appendChild(row);
  });
}

function renderDocumentFilter() {
  const select = $('#doc-filter');
  const current = select.value;
  select.innerHTML = '<option value="">全库检索</option>';
  state.docs
    .filter((doc) => doc.status === 'ready')
    .forEach((doc) => {
      const option = document.createElement('option');
      option.value = doc.id;
      option.textContent = doc.name;
      select.appendChild(option);
    });
  if (current && state.docs.some((doc) => doc.id === current)) select.value = current;
}

async function reindexDocument(doc) {
  try {
    toast(`正在重建《${doc.name}》…`);
    await api(`/api/v1/documents/${doc.id}/reindex`, { method: 'POST' });
    await loadDocuments();
    await loadSystem();
    toast('索引已更新');
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function deleteDocument(doc) {
  const confirmed = await confirmDialog('删除文档', `《${doc.name}》及其 ${doc.chunk_count || 0} 个片段会从索引中移除，原始上传文件保留在 data/uploads。`, '删除');
  if (!confirmed) return;
  try {
    await api(`/api/v1/documents/${doc.id}`, { method: 'DELETE' });
    await Promise.all([loadDocuments(), loadSystem()]);
    toast('文档已删除');
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function uploadFiles(files) {
  const list = Array.from(files || []);
  if (!list.length) return;
  const form = new FormData();
  list.forEach((file) => form.append('files', file, file.name));
  toast(`正在解析 ${list.length} 个文件…`);
  $('#upload-trigger').disabled = true;
  try {
    const response = await fetch('/api/v1/documents/upload', {
      method: 'POST',
      headers: authHeaders(),
      body: form,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.message || '上传失败');
    const successes = data.results || [];
    const failures = data.errors || [];
    successes.forEach((item) => {
      const label = item.status === 'unchanged' ? '内容未变化，已跳过' : `${item.chunk_count} 个片段`;
      toast(`《${item.name}》${label}`);
    });
    failures.forEach((item) => toast(`${item.name}: ${item.error}`, 'error'));
    if (!successes.length && !failures.length) toast('没有可处理的文件', 'error');
    await Promise.all([loadDocuments(), loadSystem()]);
    $('#doc-filter').value = '';
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    $('#upload-trigger').disabled = false;
    $('#file-input').value = '';
  }
}

async function repairIndex() {
  try {
    toast('正在修复向量索引…');
    const data = await api('/api/v1/system/index/repair', { method: 'POST' });
    toast(`已重建 ${data.repaired_vectors} 个向量`);
    await Promise.all([loadSystem(), loadDocuments()]);
  } catch (error) {
    toast(error.message, 'error');
  }
}

/* -------------------------------------------------------------------- eval */

async function loadDatasets() {
  try {
    const data = await api('/api/v1/eval/datasets');
    const select = $('#dataset-select');
    select.innerHTML = '';
    (data.datasets || []).forEach((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.name;
      option.textContent = `${dataset.name} · ${dataset.count} 题`;
      select.appendChild(option);
    });
    if (!(data.datasets || []).length) {
      select.innerHTML = '<option value="">暂无评测集</option>';
    }
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function loadEvalRuns() {
  try {
    const data = await api('/api/v1/eval/runs?limit=12');
    const runs = data.runs || [];
    renderEvalHistory(runs);
    const latest = runs.find((run) => run.status === 'completed');
    if (latest && !state.evalRun) {
      const detail = await api(`/api/v1/eval/runs/${latest.id}`);
      renderEvalRun(detail.run);
    }
  } catch {
    /* non-critical */
  }
}

function renderEvalHistory(runs) {
  const wrap = $('#eval-history');
  wrap.innerHTML = '';
  if (!runs.length) {
    wrap.innerHTML = '<div class="session-empty">还没有评测记录</div>';
    return;
  }
  const statusLabels = { completed: '已完成', running: '运行中', failed: '失败', queued: '排队中' };
  runs.forEach((run) => {
    const item = document.createElement('button');
    item.className = 'history-item';
    const hit = run.metrics?.['hit@5'];
    item.innerHTML = `<strong><span>${escapeHtml(run.dataset)} <em>${statusLabels[run.status] || run.status}</em></span><span>${hit !== undefined ? `Hit ${formatMetric(hit)}` : ''}</span></strong><span>${formatTime(run.created_at)} · ${run.total} 题</span>`;
    item.addEventListener('click', async () => {
      try {
        const detail = await api(`/api/v1/eval/runs/${run.id}`);
        renderEvalRun(detail.run);
      } catch (error) {
        toast(error.message, 'error');
      }
    });
    wrap.appendChild(item);
  });
}

function renderEvalRun(run) {
  if (!run) return;
  state.evalRun = run;
  const metrics = run.metrics || {};
  const setMetric = (selector, value, formatter = (v) => formatMetric(v)) => {
    const el = $(selector);
    el.textContent = value === undefined || value === null ? '—' : formatter(value);
  };
  setMetric('#metric-hit', metrics['hit@5']);
  setMetric('#metric-mrr', metrics['mrr@5']);
  setMetric('#metric-faith', metrics.faithfulness);
  setMetric('#metric-latency', metrics.latency_p95_ms, (v) => `${Math.round(v)}ms`);
  $('#eval-run-label').textContent = `${run.dataset} · ${formatTime(run.created_at)}`;
  const status = $('#eval-run-status');
  status.textContent = run.status === 'completed' ? '已完成' : run.status === 'running' ? '运行中' : run.status;
  status.className = `pill ${run.status === 'completed' ? '' : 'muted'}`;
  renderEvalCases(run.cases || []);
  if (run.status === 'running') {
    $('#eval-status').hidden = false;
    const percent = run.total ? Math.round((run.progress / run.total) * 100) : 0;
    $('#eval-progress-bar').style.width = `${percent}%`;
    $('#eval-status-text').textContent = `正在评测… ${run.progress}/${run.total}`;
  } else {
    $('#eval-status').hidden = true;
  }
}

function renderEvalCases(cases) {
  const tbody = $('#case-tbody');
  tbody.innerHTML = '';
  $('#eval-empty').hidden = cases.length > 0;
  state.evalCases = cases;
  cases.forEach((item, index) => {
    const metrics = item.metrics || {};
    const row = document.createElement('tr');
    row.className = 'case-row';
    row.dataset.index = index;
    row.innerHTML = `
      <td class="case-question"><strong>${escapeHtml(item.question || '')}</strong><span>${escapeHtml((item.sources || []).map((s) => s.doc_name).filter(Boolean).slice(0, 2).join(' · ') || '无来源')}</span></td>
      <td class="${(metrics['hit@5'] || 0) >= 0.5 ? 'metric-good' : 'metric-warn'}">${formatNumber(metrics['hit@5'])}</td>
      <td>${formatNumber(metrics['mrr@5'])}</td>
      <td class="${(metrics.faithfulness || 0) >= 0.7 ? 'metric-good' : 'metric-warn'}">${formatNumber(metrics.faithfulness)}</td>
      <td>${formatNumber(item.confidence)}</td>
      <td>${formatMs(metrics.latency_ms)}</td>`;
    row.addEventListener('click', () => toggleCaseDetail(index));
    tbody.appendChild(row);
  });
}

function toggleCaseDetail(index) {
  const existing = $(`tr.case-detail-row[data-index="${index}"]`);
  if (existing) {
    existing.remove();
    return;
  }
  $$('.case-detail-row').forEach((row) => row.remove());
  const item = (state.evalCases || [])[index];
  if (!item) return;
  const sources = (item.sources || []).map((s) => `${s.doc_name}${s.page ? ` · 第 ${s.page} 页` : ''}`).join(' / ');
  const row = document.createElement('tr');
  row.className = 'case-detail-row';
  row.dataset.index = index;
  row.innerHTML = `<td colspan="6">
    <div class="case-answer"><strong>回答</strong><br>${escapeHtml(item.answer || '（空）')}</div>
    <div class="trace-step-detail">来源：${escapeHtml(sources || '无')} · 置信度 ${formatNumber(item.confidence)} · ${item.retry_used ? '触发二次检索' : '单轮'}</div>
  </td>`;
  const anchor = $(`.case-row[data-index="${index}"]`);
  anchor?.after(row);
}

async function runEval() {
  const dataset = $('#dataset-select').value;
  if (!dataset) {
    toast('没有可用的评测集', 'error');
    return;
  }
  const limit = Number($('#eval-limit').value || 12);
  $('#run-eval').disabled = true;
  $('#eval-status').hidden = false;
  $('#eval-progress-bar').style.width = '4%';
  $('#eval-status-text').textContent = '正在创建评测任务…';
  clearInterval(state.evalTimer);
  try {
    const data = await api('/api/v1/eval/runs', { method: 'POST', body: { dataset, limit } });
    let run = data.run;
    renderEvalRun(run);
    const started = Date.now();
    state.evalTimer = setInterval(async () => {
      try {
        const detail = await api(`/api/v1/eval/runs/${run.id}`);
        run = detail.run;
        renderEvalRun(run);
        if (run.status !== 'running' || Date.now() - started > 180000) {
          clearInterval(state.evalTimer);
          $('#run-eval').disabled = false;
          await loadEvalRuns();
          toast(run.status === 'completed' ? '评测完成' : '评测失败', run.status === 'completed' ? '' : 'error');
        }
      } catch (error) {
        clearInterval(state.evalTimer);
        $('#run-eval').disabled = false;
        toast(error.message, 'error');
      }
    }, 750);
  } catch (error) {
    $('#run-eval').disabled = false;
    $('#eval-status').hidden = true;
    toast(error.message, 'error');
  }
}

/* ------------------------------------------------------------------ system */

function renderSystemView(data) {
  if (!data) return;
  const { llm, index, cache, retrieval, features, limits } = data;
  $('#sys-mode').textContent = llm.mock_mode ? '离线模式' : '在线模型';
  const rows = [
    ['生成模型', llm.mock_mode ? 'local-extractive' : llm.model],
    ['Provider', llm.provider],
    ['Embedding', `${llm.embedding} · ${llm.embedding_dim}d`],
    ['向量后端', `${index.vectors.backend} · ${index.vectors.count} 向量`],
    ['BM25', `v${index.bm25.version} · ${index.bm25.count} 片段`],
    ['OCR / VLM', `${llm.vision_status?.rapidocr || 'lazy'} / ${llm.vision_status?.vlm_enabled ? 'on' : 'off'}`],
    ['缓存', `精确 ${cache.exact_entries} · 语义 ${cache.semantic_entries} · 命中 ${cache.exact_hits + cache.semantic_hits}`],
  ];
  $('#sys-model-list').innerHTML = rows.map(([key, value]) => `<div class="kv-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join('');

  const labels = {
    multimodal: '多模态', ocr: 'OCR', vlm: 'VLM', incremental_index: '增量索引',
    semantic_cache: '语义缓存', evaluation: '评测', streaming: 'SSE',
  };
  $('#sys-features').innerHTML = Object.entries(labels)
    .map(([key, label]) => `<span class="feature-tag ${features[key] ? '' : 'off'}">${label}</span>`)
    .join('');

  const params = [
    ['dense_top_k', retrieval.dense_top_k], ['bm25_top_k', retrieval.bm25_top_k],
    ['rerank_top_n', retrieval.rerank_top_n], ['context_budget', retrieval.context_token_budget],
    ['min_confidence', retrieval.min_confidence], ['upload_max_mb', limits.max_upload_mb],
  ];
  $('#sys-params').innerHTML = params.map(([key, value]) => `<div class="param-row"><span>${key}</span><strong>${escapeHtml(value)}</strong></div>`).join('');
}

function applyTheme(themeId, { persist = true, notify = false } = {}) {
  const theme = THEMES.find((item) => item.id === themeId) || THEMES[0];
  document.documentElement.dataset.theme = theme.id;
  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme.id);
    } catch (error) {
      // Private mode / storage disabled: the current session still switches.
    }
  }
  const currentName = $('#system-theme-name');
  if (currentName) currentName.textContent = theme.name;
  const currentTitle = $('#theme-current-title');
  if (currentTitle) currentTitle.textContent = theme.name;
  const currentDesc = $('#theme-current-desc');
  if (currentDesc) currentDesc.textContent = theme.desc;
  const currentPill = $('#theme-current-pill');
  if (currentPill) currentPill.textContent = theme.id === 'default' ? '默认主题' : '使用中';
  $$('.theme-option').forEach((option) => {
    const active = option.dataset.themeId === theme.id;
    option.classList.toggle('active', active);
    option.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  if (notify) toast(`已切换到「${theme.name}」`);
  return theme;
}

function switchSystemTab(tab, { scroll = false, updateHash = true } = {}) {
  const next = tab === 'theme' ? 'theme' : 'intro';
  state.systemTab = next;
  $$('.system-menu-item').forEach((item) => {
    const active = item.dataset.systemTab === next;
    item.classList.toggle('active', active);
    if (active) item.setAttribute('aria-current', 'page');
    else item.removeAttribute('aria-current');
  });
  $$('.system-pane').forEach((pane) => {
    pane.classList.toggle('active', pane.dataset.systemPane === next);
  });
  if (updateHash) {
    const target = next === 'theme' ? '#system/theme' : '#system';
    if (location.hash !== target) history.replaceState(null, '', target);
  }
  if (scroll) {
    $('#view-system')?.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

/* ------------------------------------------------------------------ events */

function switchView(view, updateHash = true) {
  state.view = view;
  $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
  $$('.view').forEach((section) => section.classList.toggle('active', section.id === `view-${view}`));
  const [eyebrow, title] = TITLES[view] || TITLES.chat;
  $('#view-eyebrow').textContent = eyebrow;
  $('#view-title').textContent = title;
  $('#rail').classList.remove('open');
  const hashView = view === 'system' && state.systemTab === 'theme' ? 'system/theme' : view;
  if (updateHash && view !== 'chat' && location.hash !== `#${hashView}`) {
    history.replaceState(null, '', `#${hashView}`);
  } else if (updateHash && view === 'chat' && location.hash) {
    history.replaceState(null, '', location.pathname + location.search);
  }
  if (view === 'library') loadDocuments();
  if (view === 'eval') {
    loadDatasets();
    loadEvalRuns();
  }
  if (view === 'system') {
    loadSystem();
    switchSystemTab(state.systemTab, { updateHash: false });
  }
  if (view === 'chat') $('#message-input').focus();
  window.scrollTo({ top: 0 });
}

function bindEvents() {
  $$('.nav-item').forEach((item) => item.addEventListener('click', () => switchView(item.dataset.view)));
  $$('.system-menu-item').forEach((item) => {
    item.addEventListener('click', () => switchSystemTab(item.dataset.systemTab, { scroll: true }));
  });
  $$('.theme-option').forEach((option) => {
    option.addEventListener('click', () => applyTheme(option.dataset.themeId, { notify: true }));
  });
  window.addEventListener('hashchange', () => {
    const [view, sub] = location.hash.replace('#', '').split('/');
    if (view === 'system') state.systemTab = sub === 'theme' ? 'theme' : 'intro';
    if (TITLES[view] && view !== state.view) switchView(view, false);
    else if (view === 'system') switchSystemTab(state.systemTab, { updateHash: false });
  });
  $('#menu-toggle').addEventListener('click', () => $('#rail').classList.toggle('open'));
  $('#new-chat').addEventListener('click', newChat);
  $('#trace-toggle').addEventListener('click', toggleTrace);
  $('#trace-close').addEventListener('click', () => $('#view-chat').classList.remove('trace-open'));

  const input = $('#message-input');
  input.addEventListener('input', autoResize);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendCurrent();
    }
  });
  $('#composer').addEventListener('submit', (event) => {
    event.preventDefault();
    sendCurrent();
  });

  $('#attach-btn').addEventListener('click', () => $('#file-input').click());
  $('#file-input').addEventListener('change', (event) => uploadFiles(event.target.files));
  $('#upload-trigger').addEventListener('click', () => $('#file-input').click());

  const dropzone = $('#dropzone');
  ['dragenter', 'dragover'].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  }));
  dropzone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
  dropzone.addEventListener('click', () => $('#file-input').click());

  $('#doc-search').addEventListener('input', renderDocumentTable);
  $('#doc-status-filter').addEventListener('change', renderDocumentTable);
  $('#repair-index').addEventListener('click', repairIndex);
  $('#run-eval').addEventListener('click', runEval);

  $('#clear-cache').addEventListener('click', async () => {
    try {
      await api('/api/v1/system/cache/clear', { method: 'POST' });
      toast('缓存已清空');
      await loadSystem();
    } catch (error) {
      toast(error.message, 'error');
    }
  });

  $('#modal-backdrop').addEventListener('click', (event) => {
    if (event.target === $('#modal-backdrop')) $('#modal-cancel').click();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      $('#modal-backdrop').hidden = true;
      $('#view-chat').classList.remove('trace-open');
    }
  });

  // Citation buttons inside rendered markdown use event delegation as a
  // safety net for history messages.
  $('#chat-stream').addEventListener('click', (event) => {
    const cite = event.target.closest('[data-cite]');
    if (cite) {
      openTrace();
      highlightSource(Number(cite.dataset.cite));
    }
  });
}

/* -------------------------------------------------------------------- init */

async function init() {
  applyTheme(readThemeId(), { persist: false });
  bindEvents();
  if (window.innerWidth >= 1480) {
    $('#view-chat').classList.add('trace-open');
  }
  await Promise.allSettled([loadSystem(), loadDocuments(), loadSessions(), loadSuggestions(), loadDatasets()]);
  const [hashView, hashSub] = location.hash.replace('#', '').split('/');
  if (hashView === 'system') state.systemTab = hashSub === 'theme' ? 'theme' : 'intro';
  if (TITLES[hashView]) switchView(hashView, false);
  if (state.sessionId) {
    await openSession(state.sessionId);
  } else {
    showHero();
  }
  loadEvalRuns();
  $('#message-input').focus();

  const params = new URLSearchParams(location.search);
  state.traceAutoOpen = params.get('trace') === '1';
  const preset = params.get('q');
  if (preset && !state.streaming) {
    $('#message-input').value = preset;
    autoResize();
    setTimeout(() => sendCurrent(), 450);
  }
}

document.addEventListener('DOMContentLoaded', init);
