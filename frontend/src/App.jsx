import { useState, useRef, useEffect, useCallback, lazy, Suspense } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { MessageErrorBoundary } from './ErrorBoundary.jsx'
import { useToast } from './useToast.jsx'
import './App.css'

// Lazy-loaded components — code-split for faster initial page load
const StockChart = lazy(() => import('./StockChart.jsx'))
const NLPPanel = lazy(() => import('./NLPPanel.jsx'))
const WatchlistPanel = lazy(() => import('./WatchlistPanel.jsx'))

// ── Theme helpers ──────────────────────────────────────────────────
function getInitialTheme() {
  const saved = localStorage.getItem('theme')
  if (saved === 'light' || saved === 'dark') return saved
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme)
  localStorage.setItem('theme', theme)
}

// Apply theme immediately on load (before React renders) to avoid flash
applyTheme(getInitialTheme())

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const WELCOME_MESSAGE = {
  role: 'agent',
  content: "Hello! I'm your Financial AI Agent. Ask me about stock prices, analyst recommendations, company news, or anything finance-related.",
  timestamp: Date.now(),
}

const SUGGESTIONS = [
  "What's AAPL stock price?",
  "Compare NVDA vs AMD",
  "Latest tech sector news",
  "TSLA analyst recommendations",
]

const CONTEXTUAL_SUGGESTIONS = {
  stock: ["Show me the price chart", "What do analysts recommend?", "Show company fundamentals"],
  compare: ["Which one is a better buy?", "Show revenue comparison", "Compare P/E ratios"],
  news: ["How does this affect stock prices?", "Any related earnings reports?", "Latest SEC filings"],
}

function generateSessionId() {
  return 'sess_' + (crypto.randomUUID?.() ?? Math.random().toString(36).slice(2) + Date.now().toString(36))
}

// ── Chat history helpers ─────────────────────────────────────────
const HISTORY_KEY = 'chat_history'
const MAX_HISTORY = 50

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)))
}

/** Build a short title from the first user message in a chat */
function chatTitle(messages) {
  const first = messages.find(m => m.role === 'user')
  if (!first) return 'New chat'
  const text = first.content.trim()
  return text.length > 40 ? text.slice(0, 40) + '…' : text
}

function chatPreviewTime(messages) {
  const last = messages[messages.length - 1]
  if (!last?.timestamp) return ''
  const d = new Date(last.timestamp)
  const now = new Date()
  if (d.toDateString() === now.toDateString()) return formatTime(last.timestamp)
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function getContextualSuggestions(messages) {
  if (messages.length <= 1) return null
  const lastAgent = [...messages].reverse().find(m => m.role === 'agent' && m.content)
  if (!lastAgent) return null

  // Use NLP intent classification if available (precise), otherwise fall back to keywords
  const intent = lastAgent.nlpMeta?.intent
  if (intent === 'comparison') return CONTEXTUAL_SUGGESTIONS.compare
  if (intent === 'stock_price' || intent === 'chart' || intent === 'fundamentals') return CONTEXTUAL_SUGGESTIONS.stock
  if (intent === 'news') return CONTEXTUAL_SUGGESTIONS.news

  // Keyword fallback for messages without NLP metadata
  const content = lastAgent.content.toLowerCase()
  if (content.includes('compare') || content.includes('vs')) return CONTEXTUAL_SUGGESTIONS.compare
  if (content.includes('price') || content.includes('stock') || content.includes('$')) return CONTEXTUAL_SUGGESTIONS.stock
  if (content.includes('news') || content.includes('report')) return CONTEXTUAL_SUGGESTIONS.news
  return null
}

/** Strip internal tool-call metadata lines (e.g. "func(args) completed in 1.2s.") */
function stripToolMetadata(text) {
  return text
    // Line-anchored patterns
    .replace(/^\s*\w+\(.*?\)\s+completed\s+in\s+[\d.]+s\.?\s*$/gm, '')
    .replace(/^\s*[\u2503\u2502|]*\s*Running:\s+\w+\(.*?\)\s*$/gm, '')
    .replace(/^\s*[\u2503\u2502|]*\s*\w+\(.*?\)\s*$/gm, '')
    // Inline metadata (not on its own line)
    .replace(/\w+\([^)]*\)\s+completed\s+in\s+[\d.]+s\.?\s*/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** Format large numbers with abbreviations: 1.5B, 230M, 45K */
function formatFinancialNumber(num) {
  if (num == null || isNaN(num)) return String(num ?? '')
  const abs = Math.abs(num)
  const sign = num < 0 ? '-' : ''
  if (abs >= 1e12) return sign + (abs / 1e12).toFixed(2) + 'T'
  if (abs >= 1e9) return sign + (abs / 1e9).toFixed(2) + 'B'
  if (abs >= 1e6) return sign + (abs / 1e6).toFixed(2) + 'M'
  if (abs >= 1e3) return sign + (abs / 1e3).toFixed(1) + 'K'
  return num.toLocaleString()
}

/** Detect chart/excel filenames in agent responses and render them */
function processContent(text) {
  if (!text) return text

  // Strip tool execution metadata
  text = stripToolMetadata(text)

  const chartFiles = []
  const sheetFiles = []

  // Match chart filenames like: aapl_chart_20260220_165804.png
  let m
  const chartRe = /([a-zA-Z0-9_.-]+_chart_[a-zA-Z0-9_.-]+\.png)/g
  while ((m = chartRe.exec(text)) !== null) {
    if (!chartFiles.includes(m[1])) chartFiles.push(m[1])
  }
  // Also match bar_chart, pie_chart, line_chart, comparison, overlay filenames
  const chartRe2 = /((?:bar_chart|pie_chart|line_chart|comparison|overlay_comparison)_[a-zA-Z0-9_.-]+\.png)/g
  while ((m = chartRe2.exec(text)) !== null) {
    if (!chartFiles.includes(m[1])) chartFiles.push(m[1])
  }

  // Match Excel filenames like: aapl_report_20260220_165804.xlsx or stock_comparison_20260220.xlsx
  const sheetRe = /([a-zA-Z0-9_.-]+\.xlsx)/g
  while ((m = sheetRe.exec(text)) !== null) {
    if (!sheetFiles.includes(m[1])) sheetFiles.push(m[1])
  }

  // Append rendered markdown for charts and sheets (skip if link already present)
  for (const file of chartFiles) {
    const link = `![Stock Chart](/outputs/charts/${file})`
    if (!text.includes(link)) text += `\n\n${link}`
  }
  for (const file of sheetFiles) {
    const link = `[Download Excel File](/outputs/sheets/${file})`
    if (!text.includes(link)) text += `\n\n${link}`
  }

  return text
}

function App() {
  // Always start fresh — previous chat is auto-saved to history
  const [sessionId, setSessionId] = useState(() => {
    // Save any previous chat into history before starting fresh
    try {
      const prev = localStorage.getItem('chat_messages')
      const prevId = localStorage.getItem('chat_session_id')
      if (prev) {
        const prevMsgs = JSON.parse(prev)
        if (prevMsgs.length > 1) { // has real conversation
          const hist = loadHistory()
          // Don't duplicate if already saved
          if (!hist.some(h => h.sessionId === prevId)) {
            hist.unshift({ sessionId: prevId, messages: prevMsgs, savedAt: Date.now() })
            saveHistory(hist)
          }
        }
      }
    } catch { /* ignore */ }
    const newId = generateSessionId()
    localStorage.setItem('chat_session_id', newId)
    localStorage.removeItem('chat_messages')
    return newId
  })
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [copiedIdx, setCopiedIdx] = useState(null)
  const [showWatchlist, setShowWatchlist] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [theme, setTheme] = useState(getInitialTheme)
  const [autoScroll, setAutoScroll] = useState(true)
  const [chatHistory, setChatHistory] = useState(loadHistory)
  const [showSidebar, setShowSidebar] = useState(true)
  const [activeChatId, setActiveChatId] = useState(null) // null = current new chat
  const bottomRef = useRef(null)
  const abortRef = useRef(null)
  const streamBufferRef = useRef('')
  const flushTimerRef = useRef(null)
  const nlpMetaRef = useRef(null)
  const recognitionRef = useRef(null)
  const chatMessagesRef = useRef(null)
  const inputRef = useRef(null)
  const { addToast, ToastContainer } = useToast()

  // Persist to localStorage
  useEffect(() => {
    localStorage.setItem('chat_messages', JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    localStorage.setItem('chat_session_id', sessionId)
  }, [sessionId])

  // Smart auto-scroll: scroll to bottom only if user hasn't scrolled up
  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, loading, autoScroll])

  // Detect user scroll position to toggle auto-scroll
  useEffect(() => {
    const el = chatMessagesRef.current
    if (!el) return
    function handleScroll() {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      setAutoScroll(atBottom)
    }
    el.addEventListener('scroll', handleScroll, { passive: true })
    return () => el.removeEventListener('scroll', handleScroll)
  }, [])

  // Theme toggle
  const toggleTheme = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    applyTheme(next)
  }, [theme])

  // Global keyboard shortcuts
  useEffect(() => {
    function handleGlobalKeyDown(e) {
      // Ctrl+K / Cmd+K → focus input
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        inputRef.current?.focus()
      }
      // Escape → clear input (only when input is focused)
      if (e.key === 'Escape' && document.activeElement === inputRef.current) {
        e.preventDefault()
        setInput('')
      }
    }
    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => window.removeEventListener('keydown', handleGlobalKeyDown)
  }, [])

  /** Save current chat to history (if it has real messages) */
  const saveCurrentToHistory = useCallback(() => {
    if (messages.length <= 1) return // only welcome msg — don't save
    setChatHistory(prev => {
      // Don't duplicate
      if (prev.some(h => h.sessionId === sessionId)) return prev
      const updated = [{ sessionId, messages, savedAt: Date.now() }, ...prev]
      saveHistory(updated)
      return updated.slice(0, MAX_HISTORY)
    })
  }, [messages, sessionId])

  const handleNewChat = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    if (flushTimerRef.current) clearInterval(flushTimerRef.current)
    saveCurrentToHistory()
    const newId = generateSessionId()
    setSessionId(newId)
    setMessages([WELCOME_MESSAGE])
    setInput('')
    setLoading(false)
    setCopiedIdx(null)
    setActiveChatId(null)
    fetch(`${API_BASE}/chat/${sessionId}`, { method: 'DELETE' }).catch(() => {})
    addToast('New chat started', 'success')
  }, [sessionId, addToast, saveCurrentToHistory])

  /** Load a chat from history */
  const loadChat = useCallback((entry) => {
    // Save current chat first
    saveCurrentToHistory()
    setSessionId(entry.sessionId)
    setMessages(entry.messages)
    setActiveChatId(entry.sessionId)
    setInput('')
    setLoading(false)
    setCopiedIdx(null)
  }, [saveCurrentToHistory])

  /** Delete a chat from history */
  const deleteChat = useCallback((entryId, e) => {
    e.stopPropagation()
    setChatHistory(prev => {
      const updated = prev.filter(h => h.sessionId !== entryId)
      saveHistory(updated)
      return updated
    })
    // If viewing the deleted chat, start fresh
    if (activeChatId === entryId) {
      const newId = generateSessionId()
      setSessionId(newId)
      setMessages([WELCOME_MESSAGE])
      setActiveChatId(null)
    }
    addToast('Chat deleted', 'info')
  }, [activeChatId, addToast])

  async function copyMessage(content, idx) {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedIdx(idx)
      setTimeout(() => setCopiedIdx(null), 2000)
      addToast('Copied to clipboard', 'success')
    } catch {
      addToast('Failed to copy', 'error')
    }
  }

  async function exportChatPDF() {
    try {
      // Send messages from frontend so export works even for history chats
      const exportMessages = messages
        .filter(m => m.content)
        .map(m => ({ role: m.role, content: m.content }))
      const res = await fetch(`${API_BASE}/chat/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, messages: exportMessages }),
      })
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `chat-export-${new Date().toISOString().slice(0, 10)}.pdf`
        a.click()
        URL.revokeObjectURL(url)
        addToast('PDF exported', 'success')
      } else {
        const err = await res.json().catch(() => ({}))
        addToast(err.detail || 'PDF export failed', 'error')
      }
    } catch {
      addToast('PDF export failed', 'error')
    }
  }

  const speechSupported = typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)

  function toggleVoiceInput() {
    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      return
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) return
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = false
    recognition.continuous = false
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      setInput(prev => prev ? prev + ' ' + transcript : transcript)
      setIsListening(false)
    }
    recognition.onerror = () => setIsListening(false)
    recognition.onend = () => setIsListening(false)
    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }

  async function sendMessage(text) {
    text = (text || input).trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text, timestamp: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      const agentTimestamp = Date.now()
      streamBufferRef.current = ''
      nlpMetaRef.current = null

      // Add placeholder agent message
      setMessages(prev => [...prev, { role: 'agent', content: '', timestamp: agentTimestamp }])

      // Debounced flush: update state every 80ms instead of per-chunk
      flushTimerRef.current = setInterval(() => {
        const buffered = streamBufferRef.current
        if (buffered) {
          setMessages(prev => {
            const updated = [...prev]
            updated[updated.length - 1] = { role: 'agent', content: buffered, timestamp: agentTimestamp }
            return updated
          })
        }
      }, 80)

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') break
          if (data.startsWith('[NLP_META]')) {
            try { nlpMetaRef.current = JSON.parse(data.slice(10)) } catch { /* ignore */ }
            continue
          }
          if (data.startsWith('[ERROR]')) {
            streamBufferRef.current += `\n\n*Error: ${data.slice(8)}*`
            break
          }
          // JSON-decode the chunk to restore newlines that were encoded by the backend
          try {
            streamBufferRef.current += JSON.parse(data)
          } catch {
            streamBufferRef.current += data
          }
        }
      }

      // Final flush
      clearInterval(flushTimerRef.current)
      flushTimerRef.current = null
      const finalContent = streamBufferRef.current

      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'agent',
          content: finalContent || 'No response received. Please try again.',
          timestamp: agentTimestamp,
          nlpMeta: nlpMetaRef.current,
        }
        return updated
      })
    } catch (err) {
      if (flushTimerRef.current) {
        clearInterval(flushTimerRef.current)
        flushTimerRef.current = null
      }
      if (err.name === 'AbortError') return
      // Fallback to non-streaming endpoint
      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_id: sessionId }),
        })
        const data = await res.json()
        if (res.ok) {
          setMessages(prev => {
            const last = prev[prev.length - 1]
            if (last?.role === 'agent' && last.content === '') {
              return [...prev.slice(0, -1), { role: 'agent', content: data.response, timestamp: Date.now() }]
            }
            return [...prev, { role: 'agent', content: data.response, timestamp: Date.now() }]
          })
          if (data.session_id) setSessionId(data.session_id)
        } else {
          const errMsg = data.detail || 'Something went wrong. Please try again.'
          setMessages(prev => [...prev, { role: 'agent', content: `*Error: ${errMsg}*`, timestamp: Date.now() }])
          addToast(errMsg, 'error')
        }
      } catch {
        setMessages(prev => [...prev, { role: 'agent', content: '*Error: Could not reach the server.*', timestamp: Date.now() }])
        addToast('Could not reach the server', 'error')
      }
    } finally {
      setLoading(false)
      abortRef.current = null
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const showInitialSuggestions = messages.length <= 1 && !loading
  const contextSuggestions = !showInitialSuggestions && !loading ? getContextualSuggestions(messages) : null
  const lastMsgEmpty = messages[messages.length - 1]?.content === ''

  function scrollToBottom() {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    setAutoScroll(true)
  }

  return (
    <div className="app-layout">
    <ToastContainer />

    {/* ── Chat History Sidebar ── */}
    <aside className={`history-sidebar ${showSidebar ? 'open' : 'collapsed'}`}>
      <div className="history-sidebar-header">
        <span className="history-sidebar-title">{showSidebar ? 'Chat History' : ''}</span>
        <button
          className="sidebar-toggle-btn"
          onClick={() => setShowSidebar(v => !v)}
          aria-label={showSidebar ? 'Collapse sidebar' : 'Expand sidebar'}
          title={showSidebar ? 'Collapse' : 'Expand'}
        >
          {showSidebar ? '\u2039' : '\u203A'}
        </button>
      </div>
      {showSidebar && (
        <>
          <button className="history-new-chat-btn" onClick={handleNewChat}>
            + New Chat
          </button>
          <div className="history-list">
            {/* Current active chat (if it has messages) */}
            {!activeChatId && messages.length > 1 && (
              <div className="history-item active">
                <div className="history-item-content">
                  <span className="history-item-title">{chatTitle(messages)}</span>
                  <span className="history-item-time">Now</span>
                </div>
              </div>
            )}
            {chatHistory.map(entry => (
              <div
                key={entry.sessionId}
                className={`history-item ${activeChatId === entry.sessionId ? 'active' : ''}`}
                onClick={() => loadChat(entry)}
                role="button"
                tabIndex={0}
                onKeyDown={e => e.key === 'Enter' && loadChat(entry)}
              >
                <div className="history-item-content">
                  <span className="history-item-title">{chatTitle(entry.messages)}</span>
                  <span className="history-item-time">{chatPreviewTime(entry.messages)}</span>
                </div>
                <button
                  className="history-item-delete"
                  onClick={(e) => deleteChat(entry.sessionId, e)}
                  aria-label="Delete chat"
                  title="Delete"
                >
                  ×
                </button>
              </div>
            ))}
            {chatHistory.length === 0 && messages.length <= 1 && (
              <div className="history-empty">No previous chats</div>
            )}
          </div>
        </>
      )}
    </aside>

    {showWatchlist && (
      <Suspense fallback={<div className="lazy-loading">Loading watchlist...</div>}>
        <WatchlistPanel sessionId={sessionId} />
      </Suspense>
    )}
    <div className="chat-wrapper">
      <header className="chat-header">
        <div className="header-left">
          <button
            className="sidebar-toggle-btn mobile-only"
            onClick={() => setShowSidebar(v => !v)}
            aria-label="Toggle chat history"
            title="Chat history"
          >
            ☰
          </button>
          <span className="header-icon">📈</span>
          <span>Financial AI Agent</span>
        </div>
        <div className="header-actions">
          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? '\u2600' : '\u263E'}
          </button>
          <button
            className={`watchlist-toggle-btn ${showWatchlist ? 'active' : ''}`}
            onClick={() => setShowWatchlist(v => !v)}
            aria-label="Toggle watchlist"
            title="Watchlist"
          >
            Watchlist
          </button>
          {messages.length > 1 && (
            <button
              className="export-btn"
              onClick={exportChatPDF}
              aria-label="Export chat as PDF"
              title="Export as PDF"
            >
              ↓ PDF
            </button>
          )}
          <button
            className="new-chat-btn"
            onClick={handleNewChat}
            aria-label="Start new chat"
            title="New Chat"
          >
            + New Chat
          </button>
        </div>
      </header>

      <div className="chat-messages" ref={chatMessagesRef} role="log" aria-live="polite" style={{ position: 'relative' }}>
        {messages.map((msg, i) => (
          <div key={i} className={`bubble-row ${msg.role}`}>
            <div className={`bubble ${msg.role}`}>
              {msg.role === 'agent' && i > 0 && (
                <button
                  className={`copy-btn ${copiedIdx === i ? 'copied' : ''}`}
                  onClick={() => copyMessage(msg.content, i)}
                  aria-label={copiedIdx === i ? 'Copied' : 'Copy message'}
                  title="Copy message"
                >
                  {copiedIdx === i ? '✓' : '⧉'}
                </button>
              )}
              <MessageErrorBoundary>
                {msg.role === 'agent' && msg.nlpMeta && (
                  <Suspense fallback={null}>
                    <NLPPanel meta={msg.nlpMeta} />
                  </Suspense>
                )}
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ node, ...props }) => (
                      <div className="table-scroll"><table {...props} /></div>
                    ),
                    td: ({ node, children, ...props }) => {
                      const text = String(children ?? '')
                      const num = Number(text.replace(/[,$%]/g, ''))
                      return (
                        <td {...props}>
                          {!isNaN(num) && Math.abs(num) >= 1000
                            ? formatFinancialNumber(num)
                            : children}
                        </td>
                      )
                    },
                  }}
                >
                  {processContent(msg.content)}
                </ReactMarkdown>
                {msg.role === 'agent' && (
                  <Suspense fallback={null}>
                    <StockChart content={msg.content} />
                  </Suspense>
                )}
              </MessageErrorBoundary>
              {msg.timestamp && (
                <span className="msg-time">{formatTime(msg.timestamp)}</span>
              )}
            </div>
          </div>
        ))}
        {loading && lastMsgEmpty && (
          <div className="bubble-row agent">
            <div className="bubble agent loading">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        {loading && (
          <div className="streaming-indicator" role="status" aria-label="Agent is thinking">
            <span className="dot" /><span className="dot" /><span className="dot" />
            <span className="streaming-text">Agent is thinking...</span>
          </div>
        )}
        <div ref={bottomRef} />
        {!autoScroll && (
          <button className="scroll-bottom-btn" onClick={scrollToBottom} aria-label="Scroll to bottom" title="Scroll to bottom">
            ↓
          </button>
        )}
      </div>

      {showInitialSuggestions && (
        <div className="suggestions">
          {SUGGESTIONS.map((q, i) => (
            <button key={i} className="suggestion-chip" onClick={() => sendMessage(q)} aria-label={`Ask: ${q}`}>
              {q}
            </button>
          ))}
        </div>
      )}

      {contextSuggestions && (
        <div className="suggestions">
          {contextSuggestions.map((q, i) => (
            <button key={i} className="suggestion-chip" onClick={() => sendMessage(q)} aria-label={`Ask: ${q}`}>
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="chat-input-bar">
        {speechSupported && (
          <button
            className={`mic-btn ${isListening ? 'listening' : ''}`}
            onClick={toggleVoiceInput}
            disabled={loading}
            aria-label={isListening ? 'Stop listening' : 'Voice input'}
            title={isListening ? 'Listening...' : 'Voice input'}
          >
            {isListening ? '⏹' : '🎤'}
          </button>
        )}
        <textarea
          ref={inputRef}
          className="chat-input"
          rows={1}
          placeholder="Ask about stocks, news, fundamentals…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          aria-label="Message input"
        />
        <button
          className="send-btn"
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          aria-label="Send message"
        >
          Send
        </button>
      </div>
      <div className="kbd-hint">Enter to send · Shift+Enter newline · Esc clear · Ctrl+K focus</div>
    </div>
    </div>
  )
}

export default App
