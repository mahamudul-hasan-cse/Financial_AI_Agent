import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

function ComparePanel({ addToast }) {
  const [symbols, setSymbols] = useState('AAPL, MSFT')
  const [loading, setLoading] = useState(false)
  const [comparison, setComparison] = useState(null)

  async function runComparison() {
    const query = symbols
      .split(',')
      .map((part) => part.trim().toUpperCase())
      .filter(Boolean)
      .join(',')
    if (!query) return

    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/compare?symbols=${encodeURIComponent(query)}`)
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Comparison failed')
      setComparison(data)
    } catch (error) {
      addToast(error.message || 'Comparison failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="hero-panel">
      <div className="hero-copy">
        <p className="eyebrow">Retail Investor Copilot</p>
        <h1>Ask about a stock, understand what matters, and track it.</h1>
        <p>
          Structured research briefs, saved sessions, watch alerts, and company snapshots in one flow.
        </p>
      </div>

      <div className="compare-box">
        <label htmlFor="compare-input">Quick comparison</label>
        <div className="compare-actions">
          <input
            id="compare-input"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="AAPL, MSFT, NVDA"
          />
          <button className="primary-btn" disabled={loading} onClick={runComparison} type="button">
            {loading ? 'Comparing...' : 'Compare'}
          </button>
        </div>

        {comparison && (
          <div className="comparison-card">
            <div className="comparison-header">
              <div>
                <strong>{comparison.winner ? `${comparison.winner} leads` : 'Comparison ready'}</strong>
                <p>{comparison.recommendation}</p>
              </div>
            </div>

            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    {comparison.symbols.map((symbol) => <th key={symbol}>{symbol}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row) => (
                    <tr key={row.metric}>
                      <td>{row.metric}</td>
                      {comparison.symbols.map((symbol) => (
                        <td key={`${row.metric}-${symbol}`}>{row.values[symbol] || 'n/a'}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default ComparePanel
