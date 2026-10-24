import { describe, it, expect } from 'vitest'
import { shouldShowQuote, tickerCount } from '../quote.js'

const quote = { symbol: 'AAPL', price: 190.1 }
const one = { entities: [{ text: 'Apple', ticker: 'AAPL' }] }
const two = { entities: [{ ticker: 'AAPL' }, { ticker: 'MSFT' }] }

describe('shouldShowQuote', () => {
  it('shows for stock_price intent', () => {
    expect(shouldShowQuote({ intent: 'stock_price', quote }, one)).toBe(true)
  })
  it('shows when intent is comparison but only one ticker has a quote', () => {
    expect(shouldShowQuote({ intent: 'comparison', quote }, one)).toBe(true)
  })
  it('hides for a real two-ticker comparison', () => {
    expect(shouldShowQuote({ intent: 'comparison', quote }, two)).toBe(false)
  })
  it('hides without a usable quote', () => {
    expect(shouldShowQuote({ intent: 'stock_price', quote: null }, one)).toBe(false)
  })
  it('counts unique tickers', () => {
    expect(tickerCount({ entities: [{ ticker: 'AAPL' }, { ticker: 'AAPL' }] })).toBe(1)
    expect(tickerCount(undefined)).toBe(0)
  })
})
