import { lazy, Suspense } from 'react'
import { loadMarkdown } from './lazyMarkdown.js'

const Markdown = lazy(loadMarkdown)

const PNG_LINE_RE = /^[a-zA-Z0-9_.-]+\.png\s*$/
// Trailing "*Data as of … not financial advice.*" added by the backend.
const DISCLAIMER_RE = /\s*\*?\s*(Data as of [^*\n]*not financial advice\.)\s*\*?\s*$/

// Split the backend's disclaimer off so it renders as small print.
function splitDisclaimer(text) {
  const match = (text || '').match(DISCLAIMER_RE)
  if (!match) return { body: text || '', disclaimer: null }
  return { body: text.slice(0, match.index).trim(), disclaimer: match[1] }
}
// Base URL for generated files; override when the API is on another origin.
const OUTPUTS_BASE = import.meta.env.VITE_OUTPUTS_URL || '/outputs'

function ResearchCard({ content, data }) {
  const raw = (content || '').trim()

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
  const { body: displayText, disclaimer } = splitDisclaimer(textLines.join('\n').trim())

  // Structured artifacts: research_service returns plain strings, e.g. ["aapl_chart.png"]
  const artifactFilenames = (data?.artifacts || []).filter(a =>
    typeof a === 'string' ? a.endsWith('.png') : (a.type === 'chart' && a.filename)
  ).map(a => typeof a === 'string' ? a : a.filename)

  const chartImages = [...new Set([...artifactFilenames, ...pngLines])]

  if (!displayText && !disclaimer && chartImages.length === 0) return null

  return (
    <div className="research-card" aria-label="Assistant response">
      {displayText && (
        <div className="assistant-answer">
          {/* Plain text until the markdown chunk has loaded. */}
          <Suspense fallback={<p className="markdown-fallback">{displayText}</p>}>
            <Markdown>{displayText}</Markdown>
          </Suspense>
        </div>
      )}

      {chartImages.length > 0 && (
        <div className="chart-images">
          {chartImages.map(filename => (
            <img
              key={filename}
              src={`${OUTPUTS_BASE}/charts/${encodeURIComponent(filename)}`}
              alt="Generated chart"
              className="chart-img"
              onError={e => { e.currentTarget.style.display = 'none' }}
            />
          ))}
        </div>
      )}

      {disclaimer && <p className="answer-disclaimer">{disclaimer}</p>}
    </div>
  )
}

export default ResearchCard
