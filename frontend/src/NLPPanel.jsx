import { useState } from 'react'

const INTENT_COLORS = {
  stock_price:    '#00D4AA',
  chart:          '#4E9EFF',
  comparison:     '#A78BFA',
  news:           '#FBBF24',
  fundamentals:   '#F472B6',
  recommendation: '#34D399',
  excel:          '#FB923C',
  general:        '#8892A4',
}

const ENTITY_LABEL_COLORS = {
  ORG:        '#A78BFA',
  PERSON:     '#F9A8D4',
  GPE:        '#86EFAC',
  PRODUCT:    '#FDBA74',
  TICKER:     '#00D4AA',
  FIN_METRIC: '#C084FC',
  DATE_REF:   '#FCD34D',
  MONEY:      '#6EE7B7',
}

const SENTIMENT_COLORS = {
  positive:    '#00D4AA',
  negative:    '#FF6B6B',
  neutral:     '#8892A4',
  unavailable: '#4A5568',
}

function intentLabel(intent) {
  return intent.replace(/_/g, ' ')
}

function NLPPanel({ meta }) {
  const [expanded, setExpanded] = useState(false)
  if (!meta) return null

  const {
    entities          = [],
    intent            = 'general',
    intent_confidence = 0,
    intent_method     = 'keyword',
    sentiment         = {},
  } = meta

  const intentColor   = INTENT_COLORS[intent] || '#8892A4'
  const sentimentColor = SENTIMENT_COLORS[sentiment.label] || '#8892A4'
  const confPct       = intent_confidence > 0 ? Math.round(intent_confidence * 100) : null

  return (
    <div
      className={`nlp-panel ${expanded ? 'nlp-panel-expanded' : 'nlp-panel-collapsed'}`}
      aria-label="NLP Analysis"
    >
      <button
        className="nlp-panel-toggle"
        onClick={() => setExpanded(e => !e)}
        aria-expanded={expanded}
        type="button"
      >
        <span className="nlp-panel-icon">🧠</span>

        <span className="nlp-toggle-summary">
          {/* Intent pill */}
          <span
            className="nlp-badge nlp-badge-inline"
            style={{
              backgroundColor: intentColor + '20',
              color: intentColor,
              borderColor: intentColor + '50',
            }}
          >
            {intentLabel(intent)}
            {confPct && (
              <span style={{ marginLeft: 4, opacity: 0.75, fontSize: '0.78em' }}>
                {confPct}%
              </span>
            )}
          </span>

          {/* Sentiment pill */}
          {sentiment.label && sentiment.label !== 'unavailable' && (
            <span
              className="nlp-badge nlp-badge-inline"
              style={{
                backgroundColor: sentimentColor + '20',
                color: sentimentColor,
                borderColor: sentimentColor + '50',
              }}
            >
              {sentiment.label}
            </span>
          )}
        </span>

        {/* Method badge */}
        <span
          className="nlp-method-badge"
          style={{
            backgroundColor: intent_method === 'ml' ? '#7C3AED22' : '#47556922',
            color:            intent_method === 'ml' ? '#A78BFA'   : '#8892A4',
            borderColor:      intent_method === 'ml' ? '#7C3AED55' : '#47556955',
          }}
        >
          {intent_method === 'ml' ? 'ML' : 'KW'}
        </span>

        <span className={`nlp-chevron ${expanded ? 'nlp-chevron-open' : ''}`}>▸</span>
      </button>

      {expanded && (
        <div className="nlp-panel-body">
          {/* Intent + confidence bar */}
          <div className="nlp-row">
            <span className="nlp-label">Intent</span>
            <div className="confidence-bar-wrap">
              <span
                className="nlp-badge"
                style={{
                  backgroundColor: intentColor + '20',
                  color: intentColor,
                  borderColor: intentColor + '50',
                }}
              >
                {intentLabel(intent)}
              </span>
              {confPct && (
                <>
                  <div className="confidence-bar">
                    <div
                      className="confidence-bar-fill"
                      style={{ width: `${confPct}%`, background: intentColor }}
                    />
                  </div>
                  <span style={{ fontSize: 11, color: intentColor, fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
                    {confPct}%
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Entities */}
          {entities.length > 0 && (
            <div className="nlp-row nlp-row-entities">
              <span className="nlp-label">Entities</span>
              <div className="nlp-entity-list">
                {entities.map((e, i) => {
                  const lc = ENTITY_LABEL_COLORS[e.label] || '#8892A4'
                  return (
                    <span key={i} className="nlp-entity-chip">
                      <span className="nlp-entity-text">{e.text}</span>
                      {e.ticker && e.ticker !== e.text && (
                        <span className="nlp-entity-ticker">→ {e.ticker}</span>
                      )}
                      <span className="nlp-entity-label" style={{ color: lc }}>
                        {e.label}
                      </span>
                    </span>
                  )
                })}
              </div>
            </div>
          )}

          {/* Sentiment */}
          <div className="nlp-row">
            <span className="nlp-label">Sentiment</span>
            {sentiment.label && sentiment.label !== 'unavailable' ? (
              <span
                className="nlp-badge"
                style={{
                  backgroundColor: sentimentColor + '20',
                  color: sentimentColor,
                  borderColor: sentimentColor + '50',
                }}
              >
                {sentiment.label}
                <span className="nlp-compound">
                  {sentiment.compound >= 0 ? '+' : ''}{sentiment.compound?.toFixed(2)}
                </span>
              </span>
            ) : (
              <span
                className="nlp-badge"
                style={{ backgroundColor: '#47556922', color: '#4A5568', borderColor: '#47556955' }}
              >
                unavailable
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default NLPPanel
