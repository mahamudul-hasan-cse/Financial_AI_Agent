import { useEffect, useState, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const ALERT_TYPES = [
  { value: 'price_above', label: 'Price above' },
  { value: 'price_below', label: 'Price below' },
  { value: 'daily_digest', label: 'Daily digest' },
  { value: 'breaking_news', label: 'Breaking news' },
]

function WatchlistPanel({ sessionId, onOpenSnapshot, addToast }) {
  const [watchlist, setWatchlist] = useState([])
  const [alerts, setAlerts] = useState([])
  const [input, setInput] = useState('')
  const [alertForm, setAlertForm] = useState({
    symbol: '',
    alert_type: 'price_above',
    threshold: '',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchWatchlist = useCallback(async () => {
    if (!sessionId) return
    try {
      const [watchlistRes, alertsRes] = await Promise.all([
        fetch(`${API_BASE}/watchlist/${sessionId}`),
        fetch(`${API_BASE}/alerts/${sessionId}`),
      ])
      if (watchlistRes.ok) {
        const data = await watchlistRes.json()
        setWatchlist(data.watchlist || [])
      }
      if (alertsRes.ok) {
        const alertData = await alertsRes.json()
        setAlerts(alertData.alerts || [])
      }
    } catch {
      // keep watchlist optional
    }
  }, [sessionId])

  useEffect(() => {
    fetchWatchlist()
    const interval = setInterval(fetchWatchlist, 60000)
    return () => clearInterval(interval)
  }, [fetchWatchlist])

  async function addSymbol(e) {
    e.preventDefault()
    const symbol = input.trim().toUpperCase()
    if (!symbol) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, session_id: sessionId }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to add symbol')
      setWatchlist(data.watchlist || [])
      setInput('')
      setAlertForm((prev) => ({ ...prev, symbol }))
      addToast?.(`${symbol} added to watchlist`, 'success')
    } catch (err) {
      setError(err.message || 'Failed to add symbol')
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
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to remove symbol')
      setWatchlist(data.watchlist || [])
      addToast?.(`${symbol} removed`, 'info')
    } catch (err) {
      setError(err.message || 'Failed to remove symbol')
    }
  }

  async function createAlert(e) {
    e.preventDefault()
    const symbol = alertForm.symbol.trim().toUpperCase()
    if (!symbol) return
    try {
      const payload = {
        session_id: sessionId,
        symbol,
        alert_type: alertForm.alert_type,
      }
      if (alertForm.alert_type.startsWith('price_') && alertForm.threshold) {
        payload.threshold = Number(alertForm.threshold)
      }
      const res = await fetch(`${API_BASE}/alerts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to create alert')
      setAlerts((prev) => [data, ...prev])
      setAlertForm({ symbol, alert_type: 'price_above', threshold: '' })
      addToast?.('Alert created', 'success')
    } catch (err) {
      setError(err.message || 'Failed to create alert')
    }
  }

  async function removeAlert(id) {
    try {
      await fetch(`${API_BASE}/alerts/item/${id}`, { method: 'DELETE' })
      setAlerts((prev) => prev.filter((alert) => alert.id !== id))
      addToast?.('Alert removed', 'info')
    } catch {
      setError('Failed to remove alert')
    }
  }

  return (
    <aside className="watchlist-shell">
      <section className="watchlist-card">
        <div className="panel-header-row">
          <div>
            <p className="eyebrow">Track</p>
            <h3>Watchlist</h3>
          </div>
          <button className="ghost-btn" onClick={fetchWatchlist} type="button">
            Refresh
          </button>
        </div>

        <form className="inline-form" onSubmit={addSymbol}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Add ticker"
            maxLength={10}
          />
          <button className="primary-btn" disabled={loading || !input.trim()} type="submit">
            Add
          </button>
        </form>

        {error && <div className="inline-error">{error}</div>}

        <div className="watchlist-list">
          {watchlist.length === 0 ? (
            <div className="empty-note">Add a few names to track price moves and signals.</div>
          ) : (
            watchlist.map((item) => (
              <div className="watch-item" key={item.symbol}>
                <button className="watch-item-main" onClick={() => onOpenSnapshot?.(item.symbol)} type="button">
                  <div>
                    <strong>{item.symbol}</strong>
                    <p>{item.signal || 'No fresh signal yet'}</p>
                  </div>
                  <div className="watch-price">
                    <span>{item.price != null ? `$${item.price.toFixed(2)}` : 'n/a'}</span>
                    {item.change_pct != null && (
                      <small className={item.change_pct >= 0 ? 'tone-positive' : 'tone-negative'}>
                        {item.change_pct >= 0 ? '+' : ''}{item.change_pct.toFixed(2)}%
                      </small>
                    )}
                  </div>
                </button>
                <button className="icon-btn" onClick={() => removeSymbol(item.symbol)} type="button">
                  Remove
                </button>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="watchlist-card">
        <div className="panel-header-row">
          <div>
            <p className="eyebrow">Alerts</p>
            <h3>Simple Triggers</h3>
          </div>
        </div>

        <form className="alerts-form" onSubmit={createAlert}>
          <input
            type="text"
            value={alertForm.symbol}
            onChange={(e) => setAlertForm((prev) => ({ ...prev, symbol: e.target.value }))}
            placeholder="Ticker"
            maxLength={10}
          />
          <select
            value={alertForm.alert_type}
            onChange={(e) => setAlertForm((prev) => ({ ...prev, alert_type: e.target.value }))}
          >
            {ALERT_TYPES.map((option) => (
              <option value={option.value} key={option.value}>{option.label}</option>
            ))}
          </select>
          {alertForm.alert_type.startsWith('price_') && (
            <input
              type="number"
              step="0.01"
              value={alertForm.threshold}
              onChange={(e) => setAlertForm((prev) => ({ ...prev, threshold: e.target.value }))}
              placeholder="Threshold"
            />
          )}
          <button className="primary-btn" type="submit">Create Alert</button>
        </form>

        <div className="alert-list">
          {alerts.length === 0 ? (
            <div className="empty-note">Set threshold or news alerts for names you care about.</div>
          ) : (
            alerts.map((alert) => (
              <div className="alert-item" key={alert.id}>
                <div>
                  <strong>{alert.label}</strong>
                  <p>
                    {alert.symbol} · {alert.alert_type.replace(/_/g, ' ')}
                    {alert.threshold != null ? ` · ${alert.threshold}` : ''}
                  </p>
                </div>
                <button className="icon-btn" onClick={() => removeAlert(alert.id)} type="button">
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </section>
    </aside>
  )
}

export default WatchlistPanel
