import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ResearchCard from '../ResearchCard'
import { SIGNED_PCT_RE } from '../signedPercent'

const matches = text => [...text.matchAll(SIGNED_PCT_RE)].map(m => m[0])

describe('signed percentages', () => {
  it('matches signed changes but not ranges or unsigned values', () => {
    expect(matches('Up +1.25% today, down -0.8 % and −3% after; range 5-10%, flat 2%.'))
      .toEqual(['+1.25%', '-0.8 %', '−3%'])
  })

  it('renders gains green and losses red', async () => {
    const { container } = render(<ResearchCard content={'AAPL rose **+2.1%** while MSFT fell -0.4%.'} />)
    expect((await screen.findByText('+2.1%'))).toHaveClass('pct', 'pct-up')
    expect(container.querySelector('.pct-down')).toHaveTextContent('-0.4%')
  })

  it('leaves code untouched', async () => {
    const { container } = render(<ResearchCard content={'Use `x = -5%` here.'} />)
    await screen.findByText('x = -5%')
    expect(container.querySelector('.pct')).toBeNull()
  })
})

describe('disclaimer', () => {
  it('renders as small print, not italic body text', async () => {
    const content = 'Apple is up.\n\n*Data as of 2026-10-24 12:00 UTC. For educational purposes only — not financial advice.*'
    const { container } = render(<ResearchCard content={content} />)
    await screen.findByText('Apple is up.')
    const note = container.querySelector('.answer-disclaimer')
    expect(note).toHaveTextContent(/^Data as of 2026-10-24 12:00 UTC\. For educational purposes only — not financial advice\.$/)
    expect(container.querySelector('.assistant-answer em')).toBeNull()
  })
})
