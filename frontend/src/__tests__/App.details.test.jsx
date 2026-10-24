import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App from '../App'

function controlledStream() {
  const encoder = new TextEncoder()
  let ctrl
  const body = new ReadableStream({ start(c) { ctrl = c } })
  return {
    response: new Response(body, { status: 200 }),
    push: text => ctrl.enqueue(encoder.encode(text)),
    close: () => ctrl.close(),
  }
}

const NLP = '{"entities":[],"intent":"stock_price","intent_confidence":0.9,"intent_method":"ml","sentiment":{"label":"positive","score":0.5,"compound":0.5}}'

describe('agent bubble: skeleton and details toggle', () => {
  let stream
  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    stream = controlledStream()
    vi.stubGlobal('fetch', vi.fn(async url => (
      String(url).endsWith('/sessions')
        ? new Response(JSON.stringify({ sessions: [] }), { status: 200 })
        : stream.response
    )))
  })
  afterEach(() => vi.unstubAllGlobals())

  it('shows a typing skeleton (no pills) while waiting, then hides NLP labels behind Details', async () => {
    const { container } = render(<App />)
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'AAPL price' } })
    fireEvent.click(screen.getByLabelText('Send'))

    stream.push(`data: [NLP_META]${NLP}\n\n`)
    await screen.findByRole('status', { name: /Analyzing|Fetching|Checking|Drafting/ })
    expect(container.querySelector('.typing-skeleton')).toBeInTheDocument()
    expect(container.querySelector('.nlp-panel')).toBeNull()

    stream.push('data: [FINAL]"Apple is at $200."\n\ndata: [DONE]\n\n')
    stream.close()
    await screen.findByText('Apple is at $200.')
    expect(container.querySelector('.typing-skeleton')).toBeNull()
    expect(screen.queryByText('stock price')).not.toBeInTheDocument()

    const toggle = screen.getByRole('button', { name: 'Show details' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(toggle)
    expect(screen.getByRole('button', { name: 'Hide details' })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('stock price')).toBeInTheDocument()
    expect(screen.getByText('ML')).toBeInTheDocument()
  })

  it('puts timestamp and actions in one footer row', async () => {
    const { container } = render(<App />)
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'hi' } })
    fireEvent.click(screen.getByLabelText('Send'))
    stream.push(`data: [NLP_META]${NLP}\n\ndata: [FINAL]"Done."\n\ndata: [DONE]\n\n`)
    stream.close()
    await screen.findByText('Done.')
    const footer = container.querySelector('.agent-card .message-footer')
    expect(footer.querySelector('.message-time')).toBeInTheDocument()
    expect(footer.querySelector('[aria-label="Copy response"]')).toBeInTheDocument()
    expect(footer.querySelector('[aria-label="Show details"]')).toBeInTheDocument()
  })
})
