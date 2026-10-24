// API helpers: anonymous client identity + SSE stream parsing.

export const API_BASE = import.meta.env.VITE_API_URL || '/api'

const CLIENT_ID_KEY = 'financial-agent-client-id'

function randomId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID().replace(/-/g, '')
  const bytes = new Uint8Array(16)
  globalThis.crypto.getRandomValues(bytes)
  return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
}

// Anonymous per-browser id. The backend scopes saved sessions to it, so
// each browser only sees its own conversations. It is not a secret login —
// clearing site data starts a fresh, empty history.
export function getClientId() {
  try {
    let id = localStorage.getItem(CLIENT_ID_KEY)
    if (!id || !/^[A-Za-z0-9_-]{16,128}$/.test(id)) {
      id = `c_${randomId()}`
      localStorage.setItem(CLIENT_ID_KEY, id)
    }
    return id
  } catch {
    // Storage disabled (private mode etc.): stable for this page load only.
    getClientId.fallback ??= `c_${randomId()}`
    return getClientId.fallback
  }
}

export function generateSessionId() {
  return `sess_${randomId()}`
}

export function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {})
  headers.set('X-Client-Id', getClientId())
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  return fetch(`${API_BASE}${path}`, { ...options, headers })
}

// Incremental Server-Sent Events parser.
//
// Network reads can split an event anywhere (even mid-line), so input is
// buffered until a blank line terminates the event. Returns a function
// `push(textChunk)` that yields the `data` payload of every completed
// event; call `push('', true)` at end-of-stream to flush a trailing event.
export function createSSEParser() {
  let buffer = ''
  return function push(chunk, final = false) {
    buffer += chunk.replace(/\r\n?/g, '\n')
    const events = []
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const data = parseEvent(raw)
      if (data !== null) events.push(data)
    }
    if (final && buffer.trim()) {
      const data = parseEvent(buffer)
      buffer = ''
      if (data !== null) events.push(data)
    }
    return events
  }
}

function parseEvent(raw) {
  const dataLines = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('data:')) dataLines.push(line.slice(line.startsWith('data: ') ? 6 : 5))
  }
  return dataLines.length ? dataLines.join('\n') : null
}

// Classify one SSE data payload from /api/chat/stream.
export function interpretEvent(data) {
  if (data === '[DONE]') return { type: 'done' }
  for (const [prefix, type] of [
    ['[NLP_META]', 'nlp'],
    ['[RESEARCH_DATA]', 'research'],
    ['[FINAL]', 'final'],
  ]) {
    if (data.startsWith(prefix)) {
      try { return { type, value: JSON.parse(data.slice(prefix.length)) } }
      catch { return { type: 'ignored' } }
    }
  }
  if (data.startsWith('[STATUS]')) {
    // {"type":"status","message":"Rate limit reached — retrying in 4s…"}
    try {
      const payload = JSON.parse(data.slice(8))
      return { type: 'status', value: String(payload?.message || '') }
    } catch { return { type: 'ignored' } }
  }
  if (data.startsWith('[ERROR]')) return { type: 'error', value: data.slice(7).trim() }
  try {
    const value = JSON.parse(data)
    return typeof value === 'string' ? { type: 'text', value } : { type: 'ignored' }
  } catch {
    return { type: 'ignored' }
  }
}
