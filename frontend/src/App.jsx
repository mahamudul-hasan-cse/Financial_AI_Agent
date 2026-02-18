import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'agent', content: 'Hello! I\'m your Financial AI Agent. Ask me about stock prices, analyst recommendations, company news, or anything finance-related.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { role: 'user', content: text }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'agent', content: data.response }])
    } catch {
      setMessages(prev => [...prev, { role: 'agent', content: 'Error: could not reach the server.' }])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-wrapper">
      <header className="chat-header">
        <span className="header-icon">📈</span>
        <span>Financial AI Agent</span>
      </header>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`bubble-row ${msg.role}`}>
            <div className={`bubble ${msg.role}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        {loading && (
          <div className="bubble-row agent">
            <div className="bubble agent loading">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <textarea
          className="chat-input"
          rows={1}
          placeholder="Ask about stocks, news, fundamentals…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}

export default App
