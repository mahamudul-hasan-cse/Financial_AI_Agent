import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const PNG_LINE_RE = /^[a-zA-Z0-9_.-]+\.png\s*$/

function ResearchCard({ content, data }) {
  const raw = (content || '').trim()

  if (import.meta.env.DEV) {
    console.log('[ResearchCard] raw content:', JSON.stringify(raw.slice(0, 300)))
    console.log('[ResearchCard] artifacts:', data?.artifacts)
  }

  // Split lines, collect any standalone .png filename lines, strip them from display text
  const lines = raw.split('\n')
  const pngLines = []
  const textLines = []
  for (const line of lines) {
    if (PNG_LINE_RE.test(line.trim())) {
      pngLines.push(line.trim())
    } else {
      textLines.push(line)
    }
  }
  const displayText = textLines.join('\n').trim()

  // Structured artifacts: research_service returns plain strings, e.g. ["aapl_chart.png"]
  const artifactFilenames = (data?.artifacts || []).filter(a =>
    typeof a === 'string' ? a.endsWith('.png') : (a.type === 'chart' && a.filename)
  ).map(a => typeof a === 'string' ? a : a.filename)

  const chartImages = [...new Set([...artifactFilenames, ...pngLines])]

  if (!displayText && chartImages.length === 0) return null

  return (
    <div className="research-card" aria-label="Assistant response">
      {displayText && (
        <div className="assistant-answer">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayText}</ReactMarkdown>
        </div>
      )}

      {chartImages.length > 0 && (
        <div className="chart-images">
          {chartImages.map(filename => (
            <img
              key={filename}
              src={`/outputs/charts/${filename}`}
              alt="Generated chart"
              className="chart-img"
              onError={e => { e.currentTarget.style.display = 'none' }}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default ResearchCard
