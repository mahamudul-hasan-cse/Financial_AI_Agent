import { useEffect, useRef, useState } from 'react'
import { MessageErrorBoundary } from './ErrorBoundary.jsx'
import ResearchCard from './ResearchCard.jsx'
import NLPPanel from './NLPPanel.jsx'
import StockChart from './StockChart.jsx'
import { useToast } from './useToast.jsx'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const LOADING_STAGES = [
  'Analyzing your question…',
  'Fetching market data…',
  'Checking news and signals…',
  'Drafting the response…',
]

const STARTER_PROMPTS = [
  'What is the current price of Apple stock?',
  'Give me a quick summary of NVIDIA over the last month.',
  'Compare Microsoft and Amazon for long-term investors.',
  'What happened with Tesla recently?',
  'Summarize Google\'s fundamentals in plain language.',
  'What are the main risks for Meta stock right now?',
]

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function generateSessionId() {
  return `sess_${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2)}`
}

function getSpeechRecognitionConstructor() {
  if (typeof window === 'undefined') return null
  return window.SpeechRecognition || window.webkitSpeechRecognition || null
}

function buildSpeechText(message) {
  return message.content || ''
}

export default function App() {
  const [theme, setTheme]                   = useState(() => localStorage.getItem('theme') || 'dark')
  const [sessionId, setSessionId]           = useState(generateSessionId)
  const [messages, setMessages]             = useState([])
  const [savedSessions, setSavedSessions]   = useState([])
  const [input, setInput]                   = useState('')
  const [loading, setLoading]               = useState(false)
  const [loadingStage, setLoadingStage]     = useState(LOADING_STAGES[0])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [listening, setListening]           = useState(false)
  const [speakingId, setSpeakingId]         = useState(null)
  const [sidebarOpen, setSidebarOpen]       = useState(false)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  function toggleTheme() {
    setTheme(t => t === 'dark' ? 'light' : 'dark')
  }

  const bottomRef        = useRef(null)
  const streamBufferRef  = useRef('')
  const stageTimerRef    = useRef(null)
  const abortRef         = useRef(null)
  const recognitionRef   = useRef(null)
  const textareaRef      = useRef(null)
  const { addToast, ToastContainer } = useToast()

  useEffect(() => { loadSavedSessions() }, [])
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])
  useEffect(() => () => {
    resetStageTimer()
    abortRef.current?.abort()
    recognitionRef.current?.stop?.()
    window.speechSynthesis?.cancel()
  }, [])

  async function loadSavedSessions(activeId = null) {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (!res.ok) return
      const data = await res.json()
      setSavedSessions(data.sessions || [])
      if (activeId && !data.sessions?.some(s => s.session_id === activeId))
        setSessionId(activeId)
    } catch { /* sidebar history is optional */ }
  }

  function resetStageTimer() {
    if (stageTimerRef.current) { clearInterval(stageTimerRef.current); stageTimerRef.current = null }
  }

  function startStageTimer() {
    let idx = 0
    setLoadingStage(LOADING_STAGES[idx])
    stageTimerRef.current = setInterval(() => {
      idx = Math.min(idx + 1, LOADING_STAGES.length - 1)
      setLoadingStage(LOADING_STAGES[idx])
    }, 1200)
  }

  function handleNewChat() {
    abortRef.current?.abort()
    resetStageTimer()
    recognitionRef.current?.stop?.()
    window.speechSynthesis?.cancel()
    setListening(false); setSpeakingId(null); setMessages([])
    setInput(''); setLoading(false); setSessionId(generateSessionId()); setSidebarOpen(false)
  }

  async function handleDeleteSession(targetId, e) {
    e.stopPropagation()
    try {
      await fetch(`${API_BASE}/chat/${targetId}`, { method: 'DELETE' })
      setSavedSessions(prev => prev.filter(s => s.session_id !== targetId))
      if (targetId === sessionId) handleNewChat()
      addToast('Chat deleted', 'success')
    } catch { addToast('Could not delete chat', 'error') }
  }

  async function handleLoadSession(targetId) {
    setHistoryLoading(true)
    try {
      const res  = await fetch(`${API_BASE}/sessions/${targetId}`)
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Could not load chat history')
      setSessionId(data.session_id)
      setMessages((data.messages || []).map(msg => ({
        role: msg.role, content: msg.content,
        timestamp: msg.timestamp, nlpMeta: msg.nlp_metadata,
        structuredResponse: msg.structured_response,
      })))
      setSidebarOpen(false)
    } catch (err) {
      addToast(err.message || 'Could not load chat history', 'error')
    } finally { setHistoryLoading(false) }
  }

  function handleVoiceInput() {
    const Ctor = getSpeechRecognitionConstructor()
    if (!Ctor) { addToast('Voice input not supported in this browser.', 'info'); return }
    if (listening && recognitionRef.current) { recognitionRef.current.stop(); return }
    const rec = new Ctor()
    rec.lang = 'en-US'; rec.interimResults = true; rec.maxAlternatives = 1
    rec.onstart  = () => setListening(true)
    rec.onend    = () => setListening(false)
    rec.onerror  = () => { setListening(false); addToast('Voice input interrupted.', 'error') }
    rec.onresult = e => {
      const t = Array.from(e.results).map(r => r[0]?.transcript || '').join(' ').trim()
      setInput(t)
    }
    recognitionRef.current = rec
    rec.start()
  }

  function handleSpeakMessage(message, messageId) {
    if (!window.speechSynthesis) { addToast('Voice playback not supported.', 'info'); return }
    if (speakingId === messageId) { window.speechSynthesis.cancel(); setSpeakingId(null); return }
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(buildSpeechText(message))
    utt.rate = 1; utt.pitch = 1
    utt.onend  = () => setSpeakingId(null)
    utt.onerror = () => setSpeakingId(null)
    setSpeakingId(messageId)
    window.speechSynthesis.speak(utt)
  }

  async function sendMessage(prefilled = null) {
    const text = (prefilled ?? input).trim()
    if (!text || loading) return

    const userMsg      = { role: 'user',  content: text, timestamp: Date.now() }
    const agentTs      = Date.now() + 1
    const placeholderMsg = { role: 'agent', content: '', timestamp: agentTs }
    setMessages(prev => [...prev, userMsg, placeholderMsg])
    setInput('')
    setLoading(true)
    startStageTimer()

    const controller = new AbortController()
    abortRef.current = controller
    let finalStructured = null
    let finalNlpMeta    = null

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}))
        throw new Error(payload.detail || 'Request failed')
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      streamBufferRef.current = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })

        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') break
          if (data.startsWith('[NLP_META]')) {
            try { finalNlpMeta = JSON.parse(data.slice(10)) } catch { /* noop */ }
            continue
          }
          if (data.startsWith('[RESEARCH_DATA]')) {
            try { finalStructured = JSON.parse(data.slice(15)) } catch { /* noop */ }
            continue
          }
          if (data.startsWith('[ERROR]')) {
            streamBufferRef.current += `\n\n${data}`; continue
          }
          try       { streamBufferRef.current += JSON.parse(data) }
          catch     { streamBufferRef.current += data }

          setMessages(prev => {
            const updated = [...prev]
            updated[updated.length - 1] = {
              role: 'agent', content: streamBufferRef.current,
              timestamp: agentTs, nlpMeta: finalNlpMeta,
              structuredResponse: finalStructured,
            }
            return updated
          })
        }
      }

      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'agent',
          content: streamBufferRef.current || 'No response received. Please try again.',
          timestamp: agentTs, nlpMeta: finalNlpMeta,
          structuredResponse: finalStructured,
        }
        return updated
      })
      loadSavedSessions(sessionId)
    } catch (err) {
      if (err.name === 'AbortError') return
      try {
        const res  = await fetch(`${API_BASE}/chat`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_id: sessionId }),
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || 'Could not reach the server')
        if (data.session_id) setSessionId(data.session_id)
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            role: 'agent', content: data.response,
            timestamp: agentTs, structuredResponse: data.structured_response,
          }
          return updated
        })
        loadSavedSessions(data.session_id || sessionId)
        addToast('Recovered with fallback mode', 'info')
      } catch (fallbackErr) {
        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            role: 'agent',
            content: `Error: ${fallbackErr.message || 'Server unavailable.'}`,
            timestamp: agentTs,
          }
          return updated
        })
        addToast(fallbackErr.message || 'Server unavailable', 'error')
      }
    } finally {
      resetStageTimer(); setLoading(false); abortRef.current = null
    }
  }

  const canSend   = input.trim().length > 0 && !loading
  const emptyState = messages.length === 0

  return (
    <div className="app-shell">
      <ToastContainer />

      {/* ── Navbar ── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="navbar-logo">📈</div>
          <span className="navbar-title">Financial <span>AI</span> Agent</span>
        </div>
        <div className="navbar-right">
          <button className="theme-toggle" onClick={toggleTheme} type="button">
            <span className="theme-toggle-icon">{theme === 'dark' ? '☀️' : '🌙'}</span>
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
          <button className="ghost-btn mobile-only small-btn" onClick={() => setSidebarOpen(v => !v)} type="button">
            {sidebarOpen ? '✕ Close' : '☰ History'}
          </button>
        </div>
      </nav>

      {/* ── Body ── */}
      <div className="body-row">
        {/* Sidebar */}
        <aside className={`history-sidebar ${sidebarOpen ? 'open' : ''}`}>
          <div className="history-header">
            <p className="eyebrow">Conversations</p>
            <h2>Chat History</h2>
          </div>

          <button className="new-chat-btn" onClick={handleNewChat} type="button">
            + New Chat
          </button>

          <div className="history-list">
            {savedSessions.length === 0 ? (
              <div className="empty-note">Your conversations will appear here.</div>
            ) : (
              savedSessions.map(session => (
                <div
                  key={session.session_id}
                  className={`history-item ${session.session_id === sessionId ? 'active' : ''}`}
                >
                  <button
                    type="button"
                    className="history-item-content"
                    onClick={() => handleLoadSession(session.session_id)}
                  >
                    <strong>{session.title}</strong>
                    <p>{session.preview || 'Open to continue the conversation.'}</p>
                  </button>
                  <button
                    type="button"
                    className="history-delete-btn"
                    title="Delete chat"
                    onClick={e => handleDeleteSession(session.session_id, e)}
                  >✕</button>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* Main chat */}
        <main className="chat-layout">
          <div className="chat-stream">
            {historyLoading ? (
              <div className="history-loading">
                <span>Loading conversation…</span>
              </div>
            ) : emptyState ? (
              <div className="empty-state">
                <div className="empty-hero-icon">📊</div>
                <h2>Ask me anything about markets</h2>
                <p>Get real-time prices, analyst ratings, charts, Excel reports, and in-depth analysis — all in one place.</p>
                <div className="starter-prompts">
                  {STARTER_PROMPTS.map(prompt => (
                    <button
                      key={prompt}
                      className="starter-prompt"
                      onClick={() => sendMessage(prompt)}
                      type="button"
                    >{prompt}</button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message, idx) => {
                const msgId = `${message.timestamp}-${idx}`
                return (
                  <div className={`message-row ${message.role}`} key={msgId}>
                    {message.role === 'agent' && (
                      <div className="msg-avatar agent-avatar">🤖</div>
                    )}

                    <div className="message-bubble">
                      {message.role === 'agent' ? (
                        <div className="agent-card">
                          <div className="agent-card-toolbar">
                            <button
                              className={`tool-btn ${speakingId === msgId ? 'speaking' : ''}`}
                              onClick={() => handleSpeakMessage(message, msgId)}
                              type="button"
                            >
                              {speakingId === msgId ? '⏹ Stop' : '▶ Voice'}
                            </button>
                          </div>

                          <MessageErrorBoundary>
                            <ResearchCard content={message.content} data={message.structuredResponse} />
                          </MessageErrorBoundary>

                          {message.nlpMeta && (
                            <NLPPanel meta={message.nlpMeta} />
                          )}

                          {message.content && (
                            <div className="message-chart">
                              <StockChart content={message.content} />
                            </div>
                          )}

                          <div className="message-footer">
                            <span className="message-time">{formatTime(message.timestamp)}</span>
                          </div>
                        </div>
                      ) : (
                        <p className="plain-message">{message.content}</p>
                      )}

                      {message.role === 'user' && (
                        <span className="message-time" style={{ paddingRight: 4 }}>
                          {formatTime(message.timestamp)}
                        </span>
                      )}
                    </div>

                    {message.role === 'user' && (
                      <div className="msg-avatar user-avatar">👤</div>
                    )}
                  </div>
                )
              })
            )}
            <div ref={bottomRef} />
          </div>

          {loading && (
            <div className="loading-ribbon" role="status">
              <div className="typing-dots">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
              <span>{loadingStage}</span>
            </div>
          )}

          {/* Composer */}
          <div className="composer-wrap">
            <div className="composer">
              <div className="composer-input-row">
                <textarea
                  ref={textareaRef}
                  value={input}
                  rows={1}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault(); sendMessage()
                    }
                  }}
                  placeholder="Ask about any stock, comparison, or market…"
                />
                <div className="composer-actions">
                  <button
                    className={`voice-btn ${listening ? 'active' : ''}`}
                    onClick={handleVoiceInput}
                    type="button"
                    title={listening ? 'Stop listening' : 'Voice input'}
                  >
                    🎙
                  </button>
                  <button
                    className="send-btn"
                    disabled={!canSend}
                    onClick={() => sendMessage()}
                    type="button"
                    title="Send"
                  >
                    ➤
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
