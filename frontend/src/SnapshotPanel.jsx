import { useMemo } from 'react'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'

function SnapshotPanel({ snapshot, onClose }) {
  const chartData = useMemo(() => snapshot?.chart || [], [snapshot])

  if (!snapshot) return null

  return (
    <aside className="detail-panel">
      <div className="detail-panel-header">
        <div>
          <p className="eyebrow">Company Snapshot</p>
          <h3>{snapshot.company_name}</h3>
          <p>{snapshot.symbol}</p>
        </div>
        <button className="ghost-btn" onClick={onClose} type="button">
          Close
        </button>
      </div>

      <div className="snapshot-hero">
        <strong>{snapshot.price != null ? `$${snapshot.price.toFixed(2)}` : 'n/a'}</strong>
        <span className={snapshot.change_pct >= 0 ? 'tone-positive' : 'tone-negative'}>
          {snapshot.change_pct != null ? `${snapshot.change_pct >= 0 ? '+' : ''}${snapshot.change_pct.toFixed(2)}% today` : 'No intraday change'}
        </span>
      </div>

      {chartData.length > 1 && (
        <div className="snapshot-chart">
          <ResponsiveContainer>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="snapshotFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#d87a43" stopOpacity={0.45} />
                  <stop offset="95%" stopColor="#d87a43" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7d9c8" />
              <XAxis dataKey="label" tick={{ fontSize: 12, fill: '#765c4a' }} />
              <YAxis tick={{ fontSize: 12, fill: '#765c4a' }} />
              <Tooltip />
              <Area type="monotone" dataKey="value" stroke="#c7632f" fill="url(#snapshotFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <section className="detail-section">
        <h4>Fundamentals</h4>
        <div className="metric-grid compact">
          {snapshot.fundamentals.map((metric) => (
            <div className="metric-card tone-neutral" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="detail-section">
        <h4>Recent Headlines</h4>
        <div className="research-stack">
          {snapshot.headlines.length === 0 ? (
            <div className="empty-note">No recent headlines were available.</div>
          ) : (
            snapshot.headlines.map((headline, idx) => (
              <article className="signal-card tone-neutral" key={`${headline.title}-${idx}`}>
                <strong>{headline.title}</strong>
                <p>{headline.publisher || 'Market news source'}</p>
              </article>
            ))
          )}
        </div>
      </section>
    </aside>
  )
}

export default SnapshotPanel
