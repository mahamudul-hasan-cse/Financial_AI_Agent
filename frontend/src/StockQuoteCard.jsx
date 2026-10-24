import { useId, useState } from 'react'
import { TrendingDown, TrendingUp } from 'lucide-react'
import { hasQuote } from './quote.js'

const SPARK_W = 132
const SPARK_H = 40
const SPARK_PAD = 3

function formatMoney(value, currency) {
  try {
    return new Intl.NumberFormat(undefined, {
      style: 'currency', currency: currency || 'USD', maximumFractionDigits: 2,
    }).format(value)
  } catch {
    return `$${value.toFixed(2)}`
  }
}

function signed(value, digits = 2) {
  const abs = Math.abs(value).toFixed(digits)
  return value > 0 ? `+${abs}` : value < 0 ? `−${abs}` : abs
}

function formatDate(iso) {
  const d = new Date(`${iso}T00:00:00`)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export function Sparkline({ points, currency, trend }) {
  const [hover, setHover] = useState(null)
  const gradientId = useId()
  if (!points || points.length < 2) return null

  const closes = points.map(p => p.close)
  const min = Math.min(...closes)
  const max = Math.max(...closes)
  const span = max - min || 1
  const x = i => SPARK_PAD + (i * (SPARK_W - 2 * SPARK_PAD)) / (points.length - 1)
  const y = v => SPARK_PAD + (1 - (v - min) / span) * (SPARK_H - 2 * SPARK_PAD)
  const line = points.map((p, i) => `${x(i).toFixed(1)},${y(p.close).toFixed(1)}`).join(' ')
  const area = `${SPARK_PAD},${SPARK_H} ${line} ${SPARK_W - SPARK_PAD},${SPARK_H}`
  const first = points[0]
  const last = points[points.length - 1]
  const summary = `${formatDate(first.date)} to ${formatDate(last.date)}: ` +
    `${formatMoney(first.close, currency)} to ${formatMoney(last.close, currency)}, ` +
    `low ${formatMoney(min, currency)}, high ${formatMoney(max, currency)}`

  function handleMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    if (!rect.width) return
    const rel = ((e.clientX - rect.left) / rect.width) * SPARK_W
    const idx = Math.round(((rel - SPARK_PAD) / (SPARK_W - 2 * SPARK_PAD)) * (points.length - 1))
    setHover(Math.max(0, Math.min(points.length - 1, idx)))
  }

  const active = hover != null ? points[hover] : null

  return (
    <figure className={`sparkline sparkline-${trend}`}>
      <svg
        viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
        width={SPARK_W}
        height={SPARK_H}
        role="img"
        aria-label={`1-month closing prices, ${summary}`}
        onPointerMove={handleMove}
        onPointerLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.18" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={area} fill={`url(#${gradientId})`} />
        <polyline points={line} fill="none" stroke="currentColor" strokeWidth="2"
          strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        {active && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1="0" y2={SPARK_H} className="sparkline-crosshair" />
            <circle cx={x(hover)} cy={y(active.close)} r="3" fill="currentColor" className="sparkline-dot" />
          </>
        )}
      </svg>
      <figcaption className="sparkline-caption" aria-live="polite">
        {active
          ? `${formatDate(active.date)} · ${formatMoney(active.close, currency)}`
          : '1 month'}
      </figcaption>
    </figure>
  )
}

export default function StockQuoteCard({ quote }) {
  if (!hasQuote(quote)) return null
  const { symbol, company_name: name, price, change, change_pct: pct, currency, closes = [] } = quote
  const hasChange = typeof pct === 'number' && Number.isFinite(pct)
  const direction = !hasChange || pct === 0 ? 'flat' : pct > 0 ? 'up' : 'down'
  const points = closes.filter(p => p && typeof p.close === 'number' && Number.isFinite(p.close))
  const periodTrend = points.length >= 2
    ? (points[points.length - 1].close >= points[0].close ? 'up' : 'down')
    : direction
  const Arrow = direction === 'down' ? TrendingDown : TrendingUp

  return (
    <section className="stock-card" aria-label={`${symbol} quote`}>
      <div className="stock-card-main">
        <div className="stock-card-id">
          <span className="stock-card-symbol">{symbol}</span>
          {name && name !== symbol && <span className="stock-card-name">{name}</span>}
        </div>
        <div className="stock-card-price" data-testid="stock-price">{formatMoney(price, currency)}</div>
        {hasChange ? (
          <div className={`stock-card-change change-${direction}`} data-testid="stock-change">
            {direction !== 'flat' && <Arrow size={14} aria-hidden="true" />}
            {typeof change === 'number' && Number.isFinite(change) && <span>{signed(change)}</span>}
            <span>({signed(pct)}%)</span>
            <span className="stock-card-period">today</span>
          </div>
        ) : (
          <div className="stock-card-change change-flat">Day change unavailable</div>
        )}
      </div>
      <Sparkline points={points} currency={currency} trend={periodTrend} />
    </section>
  )
}
