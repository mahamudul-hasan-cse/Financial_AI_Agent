/**
 * NLPPanel — displays NLP preprocessing metadata alongside agent responses.
 *
 * Collapsible by default. Shows a compact summary line when collapsed,
 * and full details (intent, entities, sentiment) when expanded.
 */

import { useState } from 'react'

const INTENT_COLORS = {
  stock_price:    '#4ade80',
  chart:          '#60a5fa',
  comparison:     '#f472b6',
  news:           '#fb923c',
  fundamentals:   '#a78bfa',
  recommendation: '#34d399',
  excel:          '#fbbf24',
  general:        '#94a3b8',
}

const ENTITY_LABEL_COLORS = {
  ORG:        '#a5b4fc',
  PERSON:     '#f9a8d4',
  GPE:        '#86efac',
  PRODUCT:    '#fdba74',
  TICKER:     '#67e8f9',
  FIN_METRIC: '#c084fc',
  DATE_REF:   '#fcd34d',
  MONEY:      '#6ee7b7',
}

const SENTIMENT_COLORS = {
  positive:    '#34d399',
  negative:    '#f87171',
  neutral:     '#94a3b8',
  unavailable: '#475569',
}

function intentLabel(intent) {
  return intent.replace(/_/g, ' ')
}

function confidencePercent(confidence) {
  if (!confidence || confidence === 0) return null
  return Math.round(confidence * 100) + '%'
}

function NLPPanel({ meta }) {
  const [expanded, setExpanded] = useState(false)
  if (!meta) return null

  const {
    entities = [],
    intent = 'general',
    intent_confidence = 0,
    intent_method = 'keyword',
    sentiment = {},
  } = meta

  const intentColor = INTENT_COLORS[intent] || '#94a3b8'
  const sentimentColor = SENTIMENT_COLORS[sentiment.label] || '#94a3b8'
  const confStr = confidencePercent(intent_confidence)

  return (
    <div className={`nlp-panel ${expanded ? 'nlp-panel-expanded' : 'nlp-panel-collapsed'}`} aria-label="NLP Analysis">
      <button
        className="nlp-panel-toggle"
        onClick={() => setExpanded(e => !e)}
        aria-expanded={expanded}
        aria-label={expanded ? 'Collapse NLP analysis' : 'Expand NLP analysis'}
        type="button"
      >
        <span className="nlp-panel-icon">🧠</span>
        <span className="nlp-toggle-summary">
          <span className="nlp-badge nlp-badge-inline" style={{ backgroundColor: intentColor + '22', color: intentColor, borderColor: intentColor + '55' }}>
            {intentLabel(intent)}
            {confStr && <span style={{ marginLeft: '4px', opacity: 0.8, fontSize: '0.8em' }}>{confStr}</span>}
          </span>
          {sentiment.label && sentiment.label !== 'unavailable' && (
            <span className="nlp-badge nlp-badge-inline" style={{ backgroundColor: sentimentColor + '22', color: sentimentColor, borderColor: sentimentColor + '55' }}>
              {sentiment.label}
            </span>
          )}
        </span>
        <span className="nlp-method-badge" style={{
          backgroundColor: intent_method === 'ml' ? '#7c3aed22' : '#47556922',
          color: intent_method === 'ml' ? '#a78bfa' : '#94a3b8',
          borderColor: intent_method === 'ml' ? '#7c3aed55' : '#47556955',
        }}>
          {intent_method === 'ml' ? 'ML' : 'KW'}
        </span>
        <span className={`nlp-chevron ${expanded ? 'nlp-chevron-open' : ''}`}>▸</span>
      </button>

      {expanded && (
        <div className="nlp-panel-body">
          {/* Intent detail */}
          <div className="nlp-row">
            <span className="nlp-label">Intent</span>
            <span className="nlp-badge" style={{ backgroundColor: intentColor + '22', color: intentColor, borderColor: intentColor + '55' }}>
              {intentLabel(intent)}
              {confStr && <span style={{ marginLeft: '6px', opacity: 0.8, fontSize: '0.8em' }}>{confStr}</span>}
            </span>
          </div>

          {/* Entities */}
          {entities.length > 0 && (
            <div className="nlp-row nlp-row-entities">
              <span className="nlp-label">Entities</span>
              <div className="nlp-entity-list">
                {entities.map((e, i) => {
                  const labelColor = ENTITY_LABEL_COLORS[e.label] || '#94a3b8'
                  return (
                    <span key={i} className="nlp-entity-chip">
                      <span className="nlp-entity-text">{e.text}</span>
                      {e.ticker && e.ticker !== e.text && (
                        <span className="nlp-entity-ticker">→ {e.ticker}</span>
                      )}
                      <span className="nlp-entity-label" style={{ color: labelColor }}>
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
              <span className="nlp-badge" style={{ backgroundColor: sentimentColor + '22', color: sentimentColor, borderColor: sentimentColor + '55' }}>
                {sentiment.label}
                <span className="nlp-compound">
                  {sentiment.compound >= 0 ? '+' : ''}{sentiment.compound.toFixed(2)}
                </span>
              </span>
            ) : (
              <span className="nlp-badge" style={{ backgroundColor: '#47556922', color: '#475569', borderColor: '#47556955' }}>
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
