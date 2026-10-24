// Recharts line chart, split into its own chunk (recharts + d3 are large).
// Loaded lazily by StockChart only when a message contains price data.
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function StockChartPlot({ data: chartData }) {
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
