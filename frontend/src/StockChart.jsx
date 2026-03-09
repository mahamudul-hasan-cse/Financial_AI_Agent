import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

/**
 * Attempts to extract stock price table data from markdown content
 * and renders it as a line chart. Returns null if no parseable data found.
 *
 * Looks for markdown tables with date/price columns commonly returned
 * by the financial agent.
 */
function StockChart({ content }) {
  const chartData = useMemo(() => {
    if (!content) return null

    // Match markdown tables with date and price-like columns
    const lines = content.split('\n')
    const tableStart = lines.findIndex(l => l.includes('|') && (
      l.toLowerCase().includes('date') || l.toLowerCase().includes('price') || l.toLowerCase().includes('close')
    ))
    if (tableStart === -1) return null

    // Parse header to find relevant columns
    const headerLine = lines[tableStart]
    const headers = headerLine.split('|').map(h => h.trim().toLowerCase()).filter(Boolean)

    const dateIdx = headers.findIndex(h => h.includes('date') || h.includes('time') || h.includes('period'))
    const priceIdx = headers.findIndex(h => h.includes('price') || h.includes('close') || h.includes('value') || h.includes('open') || h.includes('high') || h.includes('change'))

    if (dateIdx === -1 || priceIdx === -1) return null

    // Skip separator line (---|---)
    const dataStartIdx = tableStart + 2
    const data = []

    for (let i = dataStartIdx; i < lines.length; i++) {
      const line = lines[i]
      if (!line.includes('|')) break

      const cols = line.split('|').map(c => c.trim()).filter(Boolean)
      if (cols.length <= Math.max(dateIdx, priceIdx)) continue

      const dateStr = cols[dateIdx].replace(/\*+/g, '').trim()
      const priceStr = cols[priceIdx].replace(/[^0-9.-]/g, '')
      const price = parseFloat(priceStr)

      if (dateStr && !isNaN(price)) {
        data.push({ date: dateStr, price })
      }
    }

    return data.length >= 2 ? data : null
  }, [content])

  if (!chartData) return null

  const minPrice = Math.min(...chartData.map(d => d.price))
  const maxPrice = Math.max(...chartData.map(d => d.price))
  const padding = (maxPrice - minPrice) * 0.1 || 1

  return (
    <div style={{ width: '100%', height: 200, margin: '0.75rem 0' }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3148" />
          <XAxis
            dataKey="date"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            stroke="#3a3f5c"
          />
          <YAxis
            domain={[minPrice - padding, maxPrice + padding]}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            stroke="#3a3f5c"
            tickFormatter={v => `$${v.toFixed(0)}`}
          />
          <Tooltip
            contentStyle={{
              background: '#1e2235',
              border: '1px solid #3a3f5c',
              borderRadius: '0.5rem',
              color: '#e2e8f0',
              fontSize: '0.85rem',
            }}
            formatter={v => [`$${v.toFixed(2)}`, 'Price']}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#818cf8"
            strokeWidth={2}
            dot={{ fill: '#818cf8', r: 3 }}
            activeDot={{ r: 5, fill: '#a5b4fc' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default StockChart
