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

describe('App streaming', () => {
  let fetchMock

  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    fetchMock = vi.fn(async (url) => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      return streamResponse([
        'data: [NLP_META]{"entities":[],"intent":"general","intent_confidence":0.5,',
        '"intent_method":"ml","sentiment":{"label":"neutral","score":0,"compound":0}}\n\ndata: "Hel',
        'lo "\n\ndata: "world"\n\n',
        'data: [FINAL]"Hello world (final)"\n\ndata: [DONE]\n\n',
      ])
    })
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => vi.unstubAllGlobals())

  it('renders split SSE chunks once, uses FINAL text, and makes a single chat request', async () => {
    render(<App />)
    fireEvent.change(screen.getByPlaceholderText(/Ask about any stock/), { target: { value: 'hi' } })
    fireEvent.click(screen.getByLabelText('Send'))

    await waitFor(() => expect(screen.getByText('Hello world (final)')).toBeInTheDocument())

    const chatCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/chat'))
    expect(chatCalls).toHaveLength(1)
    expect(String(chatCalls[0][0])).toMatch(/\/chat\/stream$/)
    const headers = chatCalls[0][1].headers
    expect(headers.get('X-Client-Id')).toMatch(/^[A-Za-z0-9_-]{16,128}$/)
    expect(JSON.parse(chatCalls[0][1].body).session_id).toMatch(/^sess_[0-9a-f]{32}$/)
  })

  it('does not fall back to a second LLM call when the stream request fails', async () => {
    fetchMock.mockImplementation(async (url) => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      return new Response(JSON.stringify({ detail: 'Rate limit exceeded.' }), {
        status: 429,
        headers: { 'Retry-After': '12' },
      })
    })
    render(<App />)
    fireEvent.change(screen.getByPlaceholderText(/Ask about any stock/), { target: { value: 'hi' } })
    fireEvent.click(screen.getByLabelText('Send'))

    await waitFor(() => expect(screen.getAllByText(/Rate limit exceeded/).length).toBeGreaterThan(0))
    const chatCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/chat'))
    expect(chatCalls).toHaveLength(1)
  })
})
