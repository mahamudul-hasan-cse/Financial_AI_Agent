import { useState, useEffect, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

function WatchlistPanel({ sessionId }) {
  const [watchlist, setWatchlist] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchWatchlist = useCallback(async () => {
    if (!sessionId) return
    try {
      const res = await fetch(`${API_BASE}/watchlist/${sessionId}`)
      if (res.ok) {
        const data = await res.json()
        setWatchlist(data.watchlist || [])
      }
    } catch {
      // silently fail — watchlist is optional
    }
  }, [sessionId])

  // Fetch on mount and every 60 seconds
  useEffect(() => {
    fetchWatchlist()
    const interval = setInterval(fetchWatchlist, 60000)
    return () => clearInterval(interval)
  }, [fetchWatchlist])

  async function addSymbol(e) {
    e.preventDefault()
    const symbol = input.trim().toUpperCase()
    if (!symbol) return
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, session_id: sessionId }),
      })
      if (res.ok) {
        const data = await res.json()
        setWatchlist(data.watchlist || [])
        setInput('')
      } else {
        const err = await res.json().catch(() => ({}))
        setError(err.detail || 'Failed to add')
      }
    } catch {
      setError('Network error')
    } finally {
      setLoading(false)
    }
  }

  async function removeSymbol(symbol) {
    try {
      const res = await fetch(`${API_BASE}/watchlist`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, session_id: sessionId }),
      })
      if (res.ok) {
        const data = await res.json()
        setWatchlist(data.watchlist || [])
      }
    } catch {
      // silently fail
    }
  }

  return (
    <div className="watchlist-panel">
      <div className="watchlist-header">
        <span className="watchlist-title">Watchlist</span>
        <button
          className="watchlist-refresh"
          onClick={fetchWatchlist}
          title="Refresh prices"
          aria-label="Refresh watchlist prices"
        >
          &#x21bb;
        </button>
      </div>

      <form className="watchlist-add-form" onSubmit={addSymbol}>
        <input
          className="watchlist-input"
          type="text"
          placeholder="Add ticker…"
          value={input}
          onChange={e => setInput(e.target.value)}
          maxLength={10}
          disabled={loading}
          aria-label="Ticker symbol"
        />
        <button
          className="watchlist-add-btn"
          type="submit"
          disabled={loading || !input.trim()}
          aria-label="Add to watchlist"
        >
          +
        </button>
      </form>
      {error && <div className="watchlist-error">{error}</div>}

      {watchlist.length === 0 ? (
        <div className="watchlist-empty">No tickers yet. Add one above.</div>
      ) : (
        <ul className="watchlist-list">
          {watchlist.map(item => (
            <li key={item.symbol} className="watchlist-item">
              <div className="watchlist-item-info">
                <span className="watchlist-symbol">{item.symbol}</span>
                <span className="watchlist-price">
                  {item.price != null ? `$${item.price.toFixed(2)}` : '—'}
                </span>
              </div>
              <div className="watchlist-item-right">
                {item.change_pct != null && (
                  <span className={`watchlist-change ${item.change_pct >= 0 ? 'positive' : 'negative'}`}>
                    {item.change_pct >= 0 ? '+' : ''}{item.change_pct.toFixed(2)}%
                  </span>
                )}
                <button
                  className="watchlist-remove"
                  onClick={() => removeSymbol(item.symbol)}
                  title={`Remove ${item.symbol}`}
                  aria-label={`Remove ${item.symbol} from watchlist`}
                >
                  &times;
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default WatchlistPanel
