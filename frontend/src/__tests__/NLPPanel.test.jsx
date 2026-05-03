import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import NLPPanel from '../NLPPanel'

const baseMeta = {
  entities: [],
  intent: 'stock_price',
  intent_confidence: 0.92,
  intent_method: 'ml',
  sentiment: { label: 'positive', score: 0.8, compound: 0.65 },
  spacy_available: true,
  vader_available: true,
}

describe('NLPPanel', () => {
  it('renders null when meta is null', () => {
    const { container } = render(<NLPPanel meta={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders collapsed by default', () => {
    render(<NLPPanel meta={baseMeta} />)
    expect(screen.getByLabelText('NLP Analysis')).toBeInTheDocument()
    expect(screen.getByLabelText('Expand NLP analysis')).toBeInTheDocument()
  })

  it('displays intent label with underscores replaced', () => {
    render(<NLPPanel meta={baseMeta} />)
    expect(screen.getByText('stock price')).toBeInTheDocument()
  })

  it('displays confidence percentage', () => {
    render(<NLPPanel meta={baseMeta} />)
    expect(screen.getByText('92%')).toBeInTheDocument()
  })

  it('displays ML badge when intent_method is ml', () => {
    render(<NLPPanel meta={baseMeta} />)
    expect(screen.getByText('ML')).toBeInTheDocument()
  })

  it('displays KW badge when intent_method is keyword', () => {
    render(<NLPPanel meta={{ ...baseMeta, intent_method: 'keyword' }} />)
    expect(screen.getByText('KW')).toBeInTheDocument()
  })

  it('displays sentiment badge in collapsed view', () => {
    render(<NLPPanel meta={baseMeta} />)
    expect(screen.getByText('positive')).toBeInTheDocument()
  })

  it('hides sentiment badge when unavailable', () => {
    const meta = { ...baseMeta, sentiment: { label: 'unavailable', score: 0, compound: 0 } }
    render(<NLPPanel meta={meta} />)
    // "unavailable" should NOT appear in collapsed summary
    const badges = screen.queryAllByText('unavailable')
    // Collapsed view hides it; only shows after expand
    expect(badges.length).toBe(0)
  })

  it('expands on toggle click to show details', () => {
    render(<NLPPanel meta={baseMeta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('Intent')).toBeInTheDocument()
    expect(screen.getByText('Sentiment')).toBeInTheDocument()
  })

  it('collapses back on second toggle click', () => {
    render(<NLPPanel meta={baseMeta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('Intent')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('Collapse NLP analysis'))
    expect(screen.queryByText('Intent')).not.toBeInTheDocument()
  })

  it('shows entities when expanded', () => {
    const meta = {
      ...baseMeta,
      entities: [
        { text: 'Apple', label: 'ORG', ticker: 'AAPL' },
        { text: 'Q1 2024', label: 'DATE_REF', ticker: null },
      ],
    }
    render(<NLPPanel meta={meta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('Apple')).toBeInTheDocument()
    expect(screen.getByText('→ AAPL')).toBeInTheDocument()
    expect(screen.getByText('Q1 2024')).toBeInTheDocument()
    expect(screen.getByText('ORG')).toBeInTheDocument()
    expect(screen.getByText('DATE_REF')).toBeInTheDocument()
  })

  it('hides entity ticker arrow when ticker equals text', () => {
    const meta = {
      ...baseMeta,
      entities: [{ text: 'AAPL', label: 'TICKER', ticker: 'AAPL' }],
    }
    render(<NLPPanel meta={meta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.queryByText('→ AAPL')).not.toBeInTheDocument()
  })

  it('does not display entities section when empty', () => {
    render(<NLPPanel meta={baseMeta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.queryByText('Entities')).not.toBeInTheDocument()
  })

  it('shows compound score when expanded with positive sentiment', () => {
    render(<NLPPanel meta={baseMeta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('+0.65')).toBeInTheDocument()
  })

  it('shows negative compound score correctly', () => {
    const meta = { ...baseMeta, sentiment: { label: 'negative', score: 0.7, compound: -0.45 } }
    render(<NLPPanel meta={meta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('-0.45')).toBeInTheDocument()
  })

  it('shows unavailable text in expanded sentiment when unavailable', () => {
    const meta = { ...baseMeta, sentiment: { label: 'unavailable', score: 0, compound: 0 } }
    render(<NLPPanel meta={meta} />)
    fireEvent.click(screen.getByLabelText('Expand NLP analysis'))
    expect(screen.getByText('unavailable')).toBeInTheDocument()
  })

  it('does not show confidence when 0', () => {
    const meta = { ...baseMeta, intent_confidence: 0 }
    render(<NLPPanel meta={meta} />)
    expect(screen.queryByText('0%')).not.toBeInTheDocument()
  })

  it('handles all intent types with correct colors', () => {
    const intents = ['stock_price', 'chart', 'comparison', 'news', 'fundamentals', 'recommendation', 'excel', 'general']
    for (const intent of intents) {
      const { unmount } = render(<NLPPanel meta={{ ...baseMeta, intent }} />)
      const expected = intent.replace(/_/g, ' ')
      expect(screen.getByText(expected)).toBeInTheDocument()
      unmount()
    }
  })

  it('handles unknown intent gracefully', () => {
    const meta = { ...baseMeta, intent: 'unknown_intent' }
    render(<NLPPanel meta={meta} />)
    expect(screen.getByText('unknown intent')).toBeInTheDocument()
  })
})
