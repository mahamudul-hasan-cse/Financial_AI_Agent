import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import App from '../App'

function streamResponse(chunks) {
  const encoder = new TextEncoder()
  const body = new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

// A stream whose next chunks are released manually, to observe in-flight UI.
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

const chatCalls = fetchMock => fetchMock.mock.calls.filter(([url]) => String(url).includes('/chat'))

async function ask(question) {
  fireEvent.change(screen.getByPlaceholderText(/Ask about any stock/), { target: { value: question } })
  fireEvent.click(screen.getByLabelText('Send'))
}

describe('App message actions and status events', () => {
  let fetchMock
  let answers

  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    answers = []
    fetchMock = vi.fn(async (url) => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      const answer = answers.shift() ?? 'Default answer'
      return streamResponse([`data: [FINAL]${JSON.stringify(answer)}\n\ndata: [DONE]\n\n`])
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => vi.unstubAllGlobals())

  it('shows retry status in the chat until text arrives', async () => {
    const stream = controlledStream()
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      return stream.response
    })
    render(<App />)
    await ask('price of AAPL')

    stream.push('data: [STATUS]{"type":"status","message":"Rate limit reached — retrying in 3s…"}\n\n')
    await waitFor(() =>
      expect(screen.getAllByText('Rate limit reached — retrying in 3s…').length).toBeGreaterThan(0),
    )
    expect(document.querySelector('.message-status')).toBeInTheDocument()

    stream.push('data: "Apple is at $200."\n\ndata: [DONE]\n\n')
    stream.close()
    await waitFor(() => expect(screen.getByText('Apple is at $200.')).toBeInTheDocument())
    expect(document.querySelector('.message-status')).not.toBeInTheDocument()
  })

  it('shows the error event when retries run out', async () => {
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      return streamResponse([
        'data: [STATUS]{"type":"status","message":"Retrying in 2s…"}\n\n',
        'data: [ERROR] The AI service is busy (rate limit reached).\n\ndata: [DONE]\n\n',
      ])
    })
    render(<App />)
    await ask('hi')
    await waitFor(() =>
      expect(screen.getByText(/Error: The AI service is busy/)).toBeInTheDocument(),
    )
    expect(document.querySelector('.message-status')).not.toBeInTheDocument()
  })

  it('copies the agent answer without chart filename lines', async () => {
    const writeText = vi.fn().mockResolvedValue()
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    answers.push('Here is the chart.\naapl_chart.png')
    render(<App />)
    await ask('chart AAPL')

    fireEvent.click(await screen.findByLabelText('Copy response'))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('Here is the chart.'))
    expect(await screen.findByText('Copied to clipboard')).toBeInTheDocument()
  })

  it('regenerate resends the previous user question', async () => {
    answers.push('First answer', 'Second answer')
    render(<App />)
    await ask('What about MSFT?')
    await screen.findByText('First answer')

    fireEvent.click(screen.getByLabelText('Regenerate response'))
    await screen.findByText('Second answer')

    const calls = chatCalls(fetchMock)
    expect(calls).toHaveLength(2)
    expect(calls.map(([, opts]) => JSON.parse(opts.body).message)).toEqual([
      'What about MSFT?',
      'What about MSFT?',
    ])
    // Both regenerate buttons now point at the same question; none while loading
    expect(screen.getAllByLabelText('Regenerate response')).toHaveLength(2)
  })

  it('hides actions on the message that is still streaming', async () => {
    const stream = controlledStream()
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      return stream.response
    })
    render(<App />)
    await ask('hi')
    stream.push('data: "partial"\n\n')
    await screen.findByText('partial')
    expect(screen.queryByLabelText('Copy response')).not.toBeInTheDocument()

    stream.push('data: [DONE]\n\n')
    stream.close()
    expect(await screen.findByLabelText('Copy response')).toBeInTheDocument()
  })
})
