import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ResearchCard from '../ResearchCard'

const mockData = {
  title: 'Apple research brief',
  summary: 'Apple remains resilient with mixed short-term sentiment.',
  symbol: 'AAPL',
  metrics: [
    { label: 'Current Price', value: '$190.25', tone: 'neutral' },
    { label: 'Today', value: '+1.20%', tone: 'positive' },
  ],
  news_signals: [
    { title: 'Services growth', detail: 'Services revenue remains a support pillar.', tone: 'positive' },
  ],
  risks: [
    { title: 'Valuation risk', detail: 'Premium multiples leave less room for disappointment.', tone: 'risk' },
  ],
  next_actions: ['Review the latest earnings transcript.'],
  sources: [{ label: 'Yahoo Finance Quote', kind: 'market_data' }],
  explainability: [{ label: 'Intent', value: 'stock_price' }],
  summary_markdown: 'Detailed markdown body',
}

describe('ResearchCard', () => {
  it('renders a readable assistant answer from structured data', () => {
    render(<ResearchCard content={'Google is trading near $173 and remains supported by cloud and AI momentum.'} data={mockData} />)

    expect(screen.getByText('Quick Take:')).toBeInTheDocument()
    expect(screen.getByText('Key Data:')).toBeInTheDocument()
    expect(screen.getByText('Risks or Cautions:')).toBeInTheDocument()
    expect(screen.getByText('Next Best Actions:')).toBeInTheDocument()
  })

  it('falls back to a structured narrative when content is missing', () => {
    render(<ResearchCard data={mockData} />)

    expect(screen.getByText('Quick Take:')).toBeInTheDocument()
    expect(screen.getByText('Key Data:')).toBeInTheDocument()
    expect(screen.getByText('Next Best Actions:')).toBeInTheDocument()
  })
})
