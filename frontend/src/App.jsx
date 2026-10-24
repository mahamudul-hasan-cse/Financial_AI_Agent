import { useEffect, useRef, useState } from 'react'
import {
  ArrowUp, Bot, ChartCandlestick, ChevronDown, Copy, History, LineChart, Mic,
  Moon, RefreshCw, Square, Sun, User, X,
} from 'lucide-react'
import { MessageErrorBoundary } from './ErrorBoundary.jsx'
import ResearchCard from './ResearchCard.jsx'
import { preloadMarkdown } from './lazyMarkdown.js'
import NLPPanel from './NLPPanel.jsx'
import Sidebar from './Sidebar.jsx'
import StockChart from './StockChart.jsx'
import StockQuoteCard from './StockQuoteCard.jsx'
import { hasQuote } from './quote.js'
import { useToast } from './useToast.jsx'
import { useSpeechInput, VOICE_LANGUAGES } from './useSpeechInput.js'
import { apiFetch, createSSEParser, generateSessionId, interpretEvent } from './api.js'
import './App.css'

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

const STORAGE = {
  theme: 'theme',
  sidebarCollapsed: 'sidebar-collapsed',
  voiceLang: 'voice-lang',
  voiceAutoSend: 'voice-autosend',
}

function readSetting(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

function writeSetting(key, value) {
  try { localStorage.setItem(key, value) } catch { /* storage disabled */ }
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// Text to put on the clipboard: the answer without standalone chart-file lines.
function copyableText(content) {
  return (content || '')
    .split('\n')
    .filter(line => !/^[a-zA-Z0-9_.-]+\.png\s*$/.test(line.trim()))
    .join('\n')
    .trim()
}

// The user question an agent message (at index idx) answered.
function questionFor(messages, idx) {
  for (let i = idx - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return messages[i].content
  }
  return null
}

const VOICE_ERRORS = {
  'not-allowed': 'Microphone permission denied. Allow mic access in your browser settings.',
  'service-not-allowed': 'Microphone permission denied. Allow mic access in your browser settings.',
  'audio-capture': 'No microphone found.',
  'no-speech': 'No speech detected. Try again.',
  'network': 'Voice recognition needs an internet connection.',
  'language-not-supported': 'This voice language is not supported by your browser.',
  'start-failed': 'Could not start voice input.',
}

function TypingSkeleton({ label }) {
  return (
    <div className="typing-skeleton" role="status" aria-label={label}>
      <div className="typing-skeleton-head">
        <span className="typing-dots" aria-hidden="true">
          <span className="typing-dot" /><span className="typing-dot" /><span className="typing-dot" />
        </span>
        <span className="typing-skeleton-label">{label}</span>
      </div>
      <span className="skeleton-line" style={{ width: '92%' }} />
      <span className="skeleton-line" style={{ width: '78%' }} />
      <span className="skeleton-line" style={{ width: '54%' }} />
    </div>
  )
}

function AgentMessage({ message, isStreaming, loading, loadingStage, canRegenerate, onCopy, onRegenerate }) {
  const [showDetails, setShowDetails] = useState(false)
  const quote = message.structuredResponse?.quote
  const showQuote = hasQuote(quote) && message.structuredResponse?.intent !== 'comparison'
  const hasDetails = !!message.nlpMeta
  const waiting = isStreaming && !message.content

  return (
    <div className="agent-card">
      {waiting && !message.status && <TypingSkeleton label={loadingStage} />}

      {message.status && !message.content && (
        <div className="message-status" role="status">{message.status}</div>
      )}

      {showQuote && <StockQuoteCard quote={quote} />}

      <MessageErrorBoundary>
        <ResearchCard content={message.content} data={message.structuredResponse} />
      </MessageErrorBoundary>

      {message.content && (
        <div className="message-chart">
          <StockChart content={message.content} />
        </div>
      )}

      {!waiting && (
        <div className="message-footer">
          <span className="message-time">{formatTime(message.timestamp)}</span>
          {message.modelInfo?.fallback && (
            <span
              className="message-model"
              title={`Primary model was busy, so ${message.modelInfo.provider} answered`}
            >via backup · {message.modelInfo.model || message.modelInfo.provider}</span>
          )}
          <div className="message-actions">
            {message.content && !isStreaming && (
              <>
                <button
                  type="button"
                  className="message-action-btn"
                  onClick={() => onCopy(message.content)}
                  aria-label="Copy response"
                  title="Copy response"
                ><Copy size={14} aria-hidden="true" /><span>Copy</span></button>
                {canRegenerate && (
                  <button
                    type="button"
                    className="message-action-btn"
                    onClick={onRegenerate}
                    disabled={loading}
                    aria-label="Regenerate response"
                    title="Ask the previous question again"
                  ><RefreshCw size={14} aria-hidden="true" /><span>Regenerate</span></button>
                )}
              </>
            )}
            {hasDetails && (
              <button
                type="button"
                className={`message-action-btn details-toggle ${showDetails ? 'open' : ''}`}
                onClick={() => setShowDetails(v => !v)}
                aria-expanded={showDetails}
                aria-label={showDetails ? 'Hide details' : 'Show details'}
                title="Intent, sentiment and entities detected for this question"
              ><span>Details</span><ChevronDown size={14} aria-hidden="true" /></button>
            )}
          </div>
        </div>
      )}

      {hasDetails && showDetails && (
        <div className="message-details">
          <NLPPanel meta={message.nlpMeta} />
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [theme, setTheme]                   = useState(() => readSetting(STORAGE.theme, 'dark'))
  const [sessionId, setSessionId]           = useState(generateSessionId)
  const [messages, setMessages]             = useState([])
  const [savedSessions, setSavedSessions]   = useState([])
  const [input, setInput]                   = useState('')
  const [loading, setLoading]               = useState(false)
  const [loadingStage, setLoadingStage]     = useState(LOADING_STAGES[0])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen]       = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => readSetting(STORAGE.sidebarCollapsed, '0') === '1')
  const [voiceLang, setVoiceLang]           = useState(() => {
    const saved = readSetting(STORAGE.voiceLang, 'en-US')
    return VOICE_LANGUAGES.some(l => l.code === saved) ? saved : 'en-US'
  })
  const [voiceAutoSend, setVoiceAutoSend]   = useState(() => readSetting(STORAGE.voiceAutoSend, '0') === '1')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    writeSetting(STORAGE.theme, theme)
  }, [theme])
  useEffect(() => { writeSetting(STORAGE.sidebarCollapsed, sidebarCollapsed ? '1' : '0') }, [sidebarCollapsed])
  useEffect(() => { writeSetting(STORAGE.voiceLang, voiceLang) }, [voiceLang])
  useEffect(() => { writeSetting(STORAGE.voiceAutoSend, voiceAutoSend ? '1' : '0') }, [voiceAutoSend])

  function toggleTheme() {
    setTheme(t => t === 'dark' ? 'light' : 'dark')
  }

  const bottomRef        = useRef(null)
  const stageTimerRef    = useRef(null)
  const abortRef         = useRef(null)
  const textareaRef      = useRef(null)
  const { addToast, ToastContainer } = useToast()

  const speech = useSpeechInput({
    lang: voiceLang,
    onTranscript: text => setInput(text),
    onFinish: (text, { error, spoke }) => {
      setInput(text)
      if (voiceAutoSend && spoke && !error && text.trim()) sendMessage(text)
      else textareaRef.current?.focus()
    },
    onError: code => {
      addToast(VOICE_ERRORS[code] || 'Voice input interrupted.', code === 'no-speech' ? 'info' : 'error')
    },
  })

  useEffect(() => { loadSavedSessions(); preloadMarkdown() }, [])
  useEffect(() => {
    // Only follow an active conversation; scrolling the empty state would
    // push its heading under the navbar on short (mobile) screens.
    if (messages.length) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])
  useEffect(() => () => {
    resetStageTimer()
    abortRef.current?.abort()
  }, [])

  async function loadSavedSessions(activeId = null) {
    try {
      const res = await apiFetch('/sessions')
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
    abortRef.current?.abort('switch')
    resetStageTimer()
    speech.abort()
    setMessages([])
    setInput(''); setLoading(false); setSessionId(generateSessionId()); setSidebarOpen(false)
  }

  async function handleDeleteSession(targetId) {
    try {
      const res = await apiFetch(`/chat/${encodeURIComponent(targetId)}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('delete failed')
      setSavedSessions(prev => prev.filter(s => s.session_id !== targetId))
      if (targetId === sessionId) handleNewChat()
      addToast('Chat deleted', 'success')
    } catch { addToast('Could not delete chat', 'error') }
  }

  async function handleRenameSession(targetId, title) {
    try {
      const res = await apiFetch(`/sessions/${encodeURIComponent(targetId)}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      })
      if (!res.ok) throw new Error('rename failed')
      const updated = await res.json()
      setSavedSessions(prev => prev.map(s => s.session_id === targetId ? { ...s, title: updated.title } : s))
    } catch { addToast('Could not rename chat', 'error') }
  }

  async function handleLoadSession(targetId) {
    abortRef.current?.abort('switch')
    speech.abort()
    setHistoryLoading(true)
    try {
      const res  = await apiFetch(`/sessions/${encodeURIComponent(targetId)}`)
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

  function handleVoiceToggle() {
    if (speech.listening) { speech.stop(); return }
    speech.start(input)
  }

  async function sendMessage(prefilled = null) {
    const text = (prefilled ?? input).trim()
    if (!text || loading) return
    // The textarea already holds the transcript; drop the live session.
    if (speech.listening) speech.abort()

    const userMsg      = { role: 'user',  content: text, timestamp: Date.now() }
    const agentTs      = Date.now() + 1
    const placeholderMsg = { role: 'agent', content: '', timestamp: agentTs }
    setMessages(prev => [...prev, userMsg, placeholderMsg])
    setInput('')
    setLoading(true)
    startStageTimer()

    const controller = new AbortController()
    abortRef.current = controller

    // One request per question: no automatic fallback to /api/chat, which
    // would run the LLM a second time (the server already retries failed
    // generations, announcing each retry with a status event).
    const state = { text: '', final: null, error: null, status: null, nlpMeta: null, structured: null, modelInfo: null }
    const render = (fallbackText = '') => {
      // The user switched/cleared the conversation: don't write into it.
      if (controller.signal.reason === 'switch') return
      setMessages(prev => {
        if (prev.length === 0) return prev
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'agent',
          content: state.final ?? (state.text || fallbackText),
          // Retry notices ("Retrying in 4s…") only matter until text arrives.
          status: state.text || state.final != null ? null : state.status,
          timestamp: agentTs,
          nlpMeta: state.nlpMeta,
          structuredResponse: state.structured,
          modelInfo: state.modelInfo,
        }
        return updated
      })
    }

    try {
      const res = await apiFetch('/chat/stream', {
        method: 'POST',
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}))
        const retryAfter = res.headers.get('Retry-After')
        const detail = payload.detail || `Request failed (${res.status})`
        const hint = retryAfter && !/try again/i.test(detail) ? ` Try again in ${retryAfter}s.` : ''
        throw new Error(res.status === 429 || res.status === 503 ? `${detail}${hint}` : detail)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      const parse   = createSSEParser()
      let finished  = false

      while (!finished) {
        const { done, value } = await reader.read()
        const events = parse(done ? decoder.decode() : decoder.decode(value, { stream: true }), done)
        for (const data of events) {
          const event = interpretEvent(data)
          if (event.type === 'done') { finished = true; break }
          if (event.type === 'nlp') state.nlpMeta = event.value
          else if (event.type === 'research') state.structured = event.value
          else if (event.type === 'final') state.final = event.value
          else if (event.type === 'model') state.modelInfo = event.value
          else if (event.type === 'error') state.error = event.value || 'Generation failed.'
          else if (event.type === 'status') {
            state.status = event.value
            resetStageTimer()
            setLoadingStage(event.value)
          }
          else if (event.type === 'text') state.text += event.value
          else continue
          render()
        }
        if (done) break
      }
      if (finished) reader.cancel().catch(() => {})

      state.status = null
      if (state.error && !state.text) {
        state.final = `Error: ${state.error}`
        addToast(state.error, 'error')
      } else if (state.error) {
        addToast('The response was interrupted. Please try again.', 'error')
      }
      render('No response received. Please try again.')
      loadSavedSessions(sessionId)
    } catch (err) {
      state.status = null
      if (controller.signal.aborted) {
        render('Stopped.')
        return
      }
      state.final = state.text
        ? `${state.text}\n\n_Connection lost: ${err.message || 'server unavailable'}._`
        : `Error: ${err.message || 'Server unavailable.'}`
      render()
      addToast(err.message || 'Server unavailable', 'error')
    } finally {
      if (abortRef.current === controller) {
        resetStageTimer(); setLoading(false); abortRef.current = null
      }
    }
  }

  function handleStop() {
    abortRef.current?.abort()
  }

  async function handleCopy(content) {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('clipboard unavailable')
      await navigator.clipboard.writeText(copyableText(content))
      addToast('Copied to clipboard', 'success')
    } catch {
      addToast('Could not copy — select the text and copy it manually.', 'error')
    }
  }

  function handleRegenerate(idx) {
    const question = questionFor(messages, idx)
    if (question) sendMessage(question)
  }

  const canSend    = input.trim().length > 0 && !loading
  const emptyState = messages.length === 0
  const listening  = speech.listening
  const micLabel   = listening ? 'Stop listening' : 'Voice input'

  return (
    <div className="app-shell">
      <ToastContainer />

      {/* ── Navbar ── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <button
            className="icon-btn mobile-only"
            onClick={() => setSidebarOpen(v => !v)}
            type="button"
            aria-label={sidebarOpen ? 'Close chat history' : 'Open chat history'}
            aria-expanded={sidebarOpen}
            title="Chat history"
          >
            {sidebarOpen ? <X size={18} aria-hidden="true" /> : <History size={18} aria-hidden="true" />}
          </button>
          <div className="navbar-logo" aria-hidden="true"><LineChart size={18} /></div>
          <span className="navbar-title">
            <span className="title-full">Financial <span className="title-accent">AI</span> Agent</span>
            <span className="title-short" aria-hidden="true">Fin<span className="title-accent">Agent</span></span>
          </span>
        </div>
        <div className="navbar-right">
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            type="button"
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <Sun size={16} aria-hidden="true" /> : <Moon size={16} aria-hidden="true" />}
            <span className="theme-toggle-label">{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
          </button>
        </div>
      </nav>

      {/* ── Body ── */}
      <div className="body-row">
        <Sidebar
          sessions={savedSessions}
          activeId={sessionId}
          collapsed={sidebarCollapsed}
          mobileOpen={sidebarOpen}
          onToggleCollapse={() => setSidebarCollapsed(v => !v)}
          onNewChat={handleNewChat}
          onSelect={handleLoadSession}
          onRename={handleRenameSession}
          onDelete={handleDeleteSession}
        />
        {sidebarOpen && (
          <div className="sidebar-backdrop mobile-only" onClick={() => setSidebarOpen(false)} aria-hidden="true" />
        )}

        {/* Main chat */}
        <main className="chat-layout">
          <div className="chat-stream">
            {historyLoading ? (
              <div className="history-loading">
                <span>Loading conversation…</span>
              </div>
            ) : emptyState ? (
              <div className="empty-state">
                <div className="empty-hero-icon" aria-hidden="true"><ChartCandlestick size={30} /></div>
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
                const isStreaming = loading && idx === messages.length - 1
                return (
                  <div className={`message-row ${message.role}`} key={msgId}>
                    {message.role === 'agent' && (
                      <div className="msg-avatar agent-avatar" aria-hidden="true"><Bot size={18} /></div>
                    )}

                    <div className="message-bubble">
                      {message.role === 'agent' ? (
                        <AgentMessage
                          message={message}
                          isStreaming={isStreaming}
                          loading={loading}
                          loadingStage={loadingStage}
                          canRegenerate={!!questionFor(messages, idx)}
                          onCopy={handleCopy}
                          onRegenerate={() => handleRegenerate(idx)}
                        />
                      ) : (
                        <>
                          <p className="plain-message">{message.content}</p>
                          <span className="message-time">{formatTime(message.timestamp)}</span>
                        </>
                      )}
                    </div>

                    {message.role === 'user' && (
                      <div className="msg-avatar user-avatar" aria-hidden="true"><User size={18} /></div>
                    )}
                  </div>
                )
              })
            )}
            <div ref={bottomRef} />
          </div>

          {/* Composer */}
          <div className="composer-wrap">
            <div className="composer">
              <div className={`composer-input-row ${listening ? 'is-listening' : ''}`}>
                <div className="composer-field">
                  {listening && (
                    <span className="listening-label" role="status">
                      <span className="listening-dot" aria-hidden="true" />Listening…
                    </span>
                  )}
                  <textarea
                    ref={textareaRef}
                    value={input}
                    rows={1}
                    readOnly={listening}
                    aria-label="Message"
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault(); sendMessage()
                      }
                    }}
                    placeholder="Ask about any stock, comparison, or market…"
                  />
                </div>
                <div className="composer-actions">
                  {speech.supported && (
                    <button
                      className={`round-btn mic-btn ${listening ? 'listening' : ''}`}
                      onClick={handleVoiceToggle}
                      type="button"
                      disabled={loading}
                      aria-pressed={listening}
                      aria-label={micLabel}
                      title={loading ? 'Voice input unavailable while answering' : micLabel}
                    >
                      <Mic size={18} aria-hidden="true" />
                    </button>
                  )}
                  {loading ? (
                    <button
                      className="round-btn send-btn"
                      onClick={handleStop}
                      type="button"
                      title="Stop generating"
                      aria-label="Stop generating"
                    >
                      <Square size={14} fill="currentColor" aria-hidden="true" />
                    </button>
                  ) : (
                    <button
                      className="round-btn send-btn"
                      disabled={!canSend}
                      onClick={() => sendMessage()}
                      type="button"
                      title="Send"
                      aria-label="Send"
                    >
                      <ArrowUp size={18} strokeWidth={2.5} aria-hidden="true" />
                    </button>
                  )}
                </div>
              </div>
              {speech.supported && (
                <div className="voice-options">
                  <label className="voice-option">
                    <span>Voice</span>
                    <select
                      value={voiceLang}
                      onChange={e => setVoiceLang(e.target.value)}
                      disabled={listening}
                      aria-label="Voice language"
                    >
                      {VOICE_LANGUAGES.map(l => (
                        <option key={l.code} value={l.code}>{l.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="voice-option">
                    <input
                      type="checkbox"
                      checked={voiceAutoSend}
                      onChange={e => setVoiceAutoSend(e.target.checked)}
                    />
                    <span>Auto-send after speaking</span>
                  </label>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
