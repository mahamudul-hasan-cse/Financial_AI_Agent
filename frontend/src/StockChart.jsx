import { lazy, Suspense, useMemo } from 'react'

const StockChartPlot = lazy(() => import('./StockChartPlot.jsx'))

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

  return (
    <Suspense fallback={<div className="stock-chart-placeholder" style={{ height: 200, margin: '0.75rem 0' }} />}>
      <StockChartPlot data={chartData} />
    </Suspense>
  )
}

export default StockChart
