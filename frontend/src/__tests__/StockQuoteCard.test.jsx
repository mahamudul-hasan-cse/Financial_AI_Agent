import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import StockQuoteCard from '../StockQuoteCard'
import App from '../App'

const QUOTE = {
  symbol: 'AAPL',
  company_name: 'Apple Inc.',
  price: 190.5,
  change: 2.5,
  change_pct: 1.33,
  currency: 'USD',
  closes: [
    { date: '2026-10-01', close: 180 },
    { date: '2026-10-02', close: 185 },
    { date: '2026-10-03', close: 190.5 },
  ],
}

describe('StockQuoteCard', () => {
  it('shows ticker, price and a green signed day change', () => {
    render(<StockQuoteCard quote={QUOTE} />)
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    expect(screen.getByTestId('stock-price')).toHaveTextContent('190.50')
    const change = screen.getByTestId('stock-change')
    expect(change).toHaveClass('change-up')
    expect(change).toHaveTextContent('+2.50')
    expect(change).toHaveTextContent('(+1.33%)')
  })

  it('marks a falling day red with a minus sign', () => {
    render(<StockQuoteCard quote={{ ...QUOTE, change: -3.1, change_pct: -1.6 }} />)
    const change = screen.getByTestId('stock-change')
    expect(change).toHaveClass('change-down')
    expect(change).toHaveTextContent('−3.10')
    expect(change).toHaveTextContent('(−1.60%)')
  })

  it('draws a sparkline from the real closes with an accessible summary', () => {
    const { container } = render(<StockQuoteCard quote={QUOTE} />)
    const svg = screen.getByRole('img', { name: /1-month closing prices/ })
    expect(svg.getAttribute('aria-label')).toMatch(/180\.00.*190\.50/)
    const points = container.querySelector('polyline').getAttribute('points').trim().split(' ')
    expect(points).toHaveLength(3)
    expect(container.querySelector('.sparkline')).toHaveClass('sparkline-up')
  })

  it('shows the hovered close in the caption', () => {
    const { container } = render(<StockQuoteCard quote={QUOTE} />)
    const svg = container.querySelector('.sparkline svg')
    svg.getBoundingClientRect = () => ({ left: 0, width: 132, top: 0, height: 40 })
    fireEvent.pointerMove(svg, { clientX: 0 })
    expect(container.querySelector('figcaption')).toHaveTextContent('180.00')
    fireEvent.pointerLeave(svg)
    expect(container.querySelector('figcaption')).toHaveTextContent('1 month')
  })

  it('omits the sparkline with fewer than two closes', () => {
    const { container } = render(<StockQuoteCard quote={{ ...QUOTE, closes: [QUOTE.closes[0]] }} />)
    expect(container.querySelector('.sparkline')).toBeNull()
    expect(screen.getByTestId('stock-price')).toBeInTheDocument()
  })

  it('renders nothing without a real price', () => {
    const { container } = render(<StockQuoteCard quote={{ ...QUOTE, price: null }} />)
    expect(container.innerHTML).toBe('')
  })

  it('says the day change is unavailable instead of inventing one', () => {
    render(<StockQuoteCard quote={{ ...QUOTE, change: null, change_pct: null }} />)
    expect(screen.getByText('Day change unavailable')).toBeInTheDocument()
  })
})

describe('App stock card', () => {
  const research = intent => JSON.stringify({ title: 't', summary: 's', intent, quote: QUOTE })

  function stubStream(intent) {
    vi.stubGlobal('fetch', vi.fn(async url => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      const body = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode(
            `data: [FINAL]"Apple trades at $190.50."\n\ndata: [RESEARCH_DATA]${research(intent)}\n\ndata: [DONE]\n\n`,
          ))
          c.close()
        },
      })
      return new Response(body, { status: 200 })
    }))
  }

  beforeEach(() => { window.HTMLElement.prototype.scrollIntoView = vi.fn() })
  afterEach(() => vi.unstubAllGlobals())

  async function ask() {
    render(<App />)
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'AAPL price' } })
    fireEvent.click(screen.getByLabelText('Send'))
    await screen.findByText('Apple trades at $190.50.')
  }

  it('renders the quote card from RESEARCH_DATA for a price answer', async () => {
    stubStream('stock_price')
    await ask()
    expect(screen.getByRole('region', { name: 'AAPL quote' })).toBeInTheDocument()
  })

  it('skips the single-ticker card for comparisons', async () => {
    stubStream('comparison')
    await ask()
    expect(screen.queryByRole('region', { name: 'AAPL quote' })).not.toBeInTheDocument()
  })
})
