import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import App from '../App'
import { SILENCE_TIMEOUT_MS, MAX_LISTEN_MS, STOP_GRACE_MS } from '../useSpeechInput'

// Minimal Web Speech API double. Tests drive it via emitResult/emitError/end.
class MockRecognition {
  static instances = []
  constructor() {
    this.lang = ''
    this.started = false
    this.start = vi.fn(() => {
      if (this.started) throw new Error('InvalidStateError: already started')
      this.started = true
    })
    this.stop = vi.fn(() => { queueMicrotask(() => this.end()) })
    this.abort = vi.fn(() => { this.started = false })
    MockRecognition.instances.push(this)
  }
  // results: [[transcript, isFinal], …] for the whole session so far
  emitResult(results, resultIndex = 0) {
    const list = results.map(([transcript, isFinal]) => {
      const r = [{ transcript }]
      r.isFinal = isFinal
      return r
    })
    this.onresult?.({ resultIndex, results: list })
  }
  emitError(error) { this.onerror?.({ error }) }
  end() { this.started = false; this.onend?.() }
}

const latest = () => MockRecognition.instances[MockRecognition.instances.length - 1]
const textarea = () => screen.getByLabelText('Message')
const micButton = () => screen.queryByRole('button', { name: /voice input|stop listening/i })

function sessionsOrAnswer(url) {
  if (String(url).endsWith('/sessions')) {
    return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
  }
  const body = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode('data: [FINAL]"Voice answer"\n\ndata: [DONE]\n\n'))
      c.close()
    },
  })
  return new Response(body, { status: 200 })
}

describe('voice input', () => {
  let fetchMock

  beforeEach(() => {
    localStorage.clear()
    MockRecognition.instances = []
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    window.SpeechRecognition = MockRecognition
    fetchMock = vi.fn(async url => sessionsOrAnswer(url))
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    delete window.SpeechRecognition
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('is hidden when the browser has no speech recognition', () => {
    delete window.SpeechRecognition
    render(<App />)
    expect(micButton()).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Voice language')).not.toBeInTheDocument()
  })

  it('shows listening state, interim text, and keeps final text after typed text', async () => {
    render(<App />)
    fireEvent.change(textarea(), { target: { value: 'Compare' } })
    const mic = micButton()
    expect(mic).toHaveAttribute('aria-pressed', 'false')
    expect(mic).toHaveAttribute('title', 'Voice input')

    fireEvent.click(mic)
    expect(latest().start).toHaveBeenCalledTimes(1)
    expect(micButton()).toHaveAttribute('aria-pressed', 'true')
    expect(micButton()).toHaveAccessibleName('Stop listening')
    expect(screen.getByText('Listening…')).toBeInTheDocument()

    act(() => latest().emitResult([['apple and', false]]))
    expect(textarea()).toHaveValue('Compare apple and')

    act(() => latest().emitResult([['apple and microsoft', true]]))
    expect(textarea()).toHaveValue('Compare apple and microsoft')

    act(() => latest().emitResult([['apple and microsoft', true], ['stock', false]]))
    expect(textarea()).toHaveValue('Compare apple and microsoft stock')

    // Stop mid-phrase: interim text is committed, not lost.
    fireEvent.click(micButton())
    await waitFor(() => expect(screen.queryByText('Listening…')).not.toBeInTheDocument())
    expect(textarea()).toHaveValue('Compare apple and microsoft stock')
    expect(micButton()).toHaveAttribute('aria-pressed', 'false')
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/chat'))).toBe(false)
  })

  it('ignores a second start while already listening', () => {
    render(<App />)
    const mic = micButton()
    // Two clicks in one batch: the second sees the session already active.
    act(() => { mic.click(); mic.click() })
    expect(MockRecognition.instances).toHaveLength(1)
    expect(latest().start).toHaveBeenCalledTimes(1)
  })

  it('stops after the silence timeout, resetting the timer on each result', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(micButton())
    const rec = latest()

    act(() => vi.advanceTimersByTime(SILENCE_TIMEOUT_MS - 1000))
    act(() => rec.emitResult([['hello', false]]))
    act(() => vi.advanceTimersByTime(SILENCE_TIMEOUT_MS - 1000))
    expect(rec.stop).not.toHaveBeenCalled()

    act(() => vi.advanceTimersByTime(1000))
    expect(rec.stop).toHaveBeenCalledTimes(1)
  })

  it('stops at the hard maximum duration even while speech keeps coming', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(micButton())
    const rec = latest()
    for (let t = 0; t < MAX_LISTEN_MS - 5000; t += 5000) {
      act(() => vi.advanceTimersByTime(5000))
      act(() => rec.emitResult([['still talking', false]]))
    }
    expect(rec.stop).not.toHaveBeenCalled()
    act(() => vi.advanceTimersByTime(5000))
    expect(rec.stop).toHaveBeenCalledTimes(1)
  })

  it('uses the selected language and persists it', () => {
    const { unmount } = render(<App />)
    fireEvent.click(micButton())
    expect(latest().lang).toBe('en-US')
    act(() => latest().end())

    fireEvent.change(screen.getByLabelText('Voice language'), { target: { value: 'bn-BD' } })
    expect(localStorage.getItem('voice-lang')).toBe('bn-BD')
    fireEvent.click(micButton())
    expect(latest().lang).toBe('bn-BD')
    unmount()

    render(<App />)
    expect(screen.getByLabelText('Voice language')).toHaveValue('bn-BD')
  })

  it('does not auto-send by default', async () => {
    render(<App />)
    expect(screen.getByLabelText(/auto-send/i)).not.toBeChecked()
    fireEvent.click(micButton())
    act(() => latest().emitResult([['price of apple', true]]))
    act(() => latest().end())
    expect(textarea()).toHaveValue('price of apple')
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/chat'))).toBe(false)
  })

  it('auto-sends the transcript when enabled (and persists the toggle)', async () => {
    render(<App />)
    fireEvent.click(screen.getByLabelText(/auto-send/i))
    expect(localStorage.getItem('voice-autosend')).toBe('1')

    fireEvent.click(micButton())
    act(() => latest().emitResult([['price of apple', true]]))
    act(() => latest().end())

    await screen.findByText('Voice answer')
    const chat = fetchMock.mock.calls.filter(([u]) => String(u).includes('/chat'))
    expect(chat).toHaveLength(1)
    expect(JSON.parse(chat[0][1].body).message).toBe('price of apple')
  })

  it('does not auto-send after an error, but keeps the text', async () => {
    localStorage.setItem('voice-autosend', '1')
    render(<App />)
    fireEvent.click(micButton())
    act(() => latest().emitResult([['partial words', false]]))
    act(() => latest().emitError('network'))
    act(() => latest().end())

    expect(await screen.findByText(/needs an internet connection/)).toBeInTheDocument()
    expect(textarea()).toHaveValue('partial words')
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/chat'))).toBe(false)
  })

  it('shows a permission error toast once and resets state', async () => {
    render(<App />)
    fireEvent.click(micButton())
    act(() => latest().emitError('not-allowed'))
    act(() => latest().end())
    expect(await screen.findAllByText(/Microphone permission denied/)).toHaveLength(1)
    expect(micButton()).toHaveAttribute('aria-pressed', 'false')
  })

  it('is silent on aborted and recovers when start() throws', () => {
    render(<App />)
    fireEvent.click(micButton())
    act(() => latest().emitError('aborted'))
    act(() => latest().end())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()

    window.SpeechRecognition = class extends MockRecognition {
      constructor() { super(); this.start = vi.fn(() => { throw new Error('boom') }) }
    }
    fireEvent.click(micButton())
    expect(screen.getByText('Could not start voice input.')).toBeInTheDocument()
    expect(micButton()).toHaveAttribute('aria-pressed', 'false')
  })

  it('is disabled while an answer is loading', async () => {
    let release
    fetchMock.mockImplementation(async url => {
      if (String(url).endsWith('/sessions')) return sessionsOrAnswer(url)
      await new Promise(r => { release = r })
      return sessionsOrAnswer(url)
    })
    render(<App />)
    fireEvent.change(textarea(), { target: { value: 'hi' } })
    fireEvent.click(screen.getByLabelText('Send'))
    await waitFor(() => expect(micButton()).toBeDisabled())
    release()
    await screen.findByText('Voice answer')
    expect(micButton()).not.toBeDisabled()
  })

  it('aborts the recognition on unmount and on new chat', () => {
    const { unmount } = render(<App />)
    fireEvent.click(micButton())
    const first = latest()
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    expect(first.abort).toHaveBeenCalled()
    expect(micButton()).toHaveAttribute('aria-pressed', 'false')

    fireEvent.click(micButton())
    const second = latest()
    expect(second).not.toBe(first)
    // A late event from the aborted session must not touch the input.
    act(() => first.emitResult([['ghost', true]]))
    expect(textarea()).toHaveValue('')

    unmount()
    expect(second.abort).toHaveBeenCalled()
  })

  it('a second stop force-finishes when the browser never fires onend', () => {
    render(<App />)
    fireEvent.click(micButton())
    const first = latest()
    first.stop.mockImplementation(() => {}) // onend never arrives
    act(() => first.emitResult([['hello', true]]))

    fireEvent.click(micButton()) // graceful stop: still winding down
    expect(first.stop).toHaveBeenCalledTimes(1)
    expect(micButton()).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(micButton()) // force
    expect(first.abort).toHaveBeenCalled()
    expect(micButton()).toHaveAttribute('aria-pressed', 'false')
    expect(textarea()).toHaveValue('hello')

    fireEvent.click(micButton()) // restart is clean, with the kept text as base
    expect(MockRecognition.instances).toHaveLength(2)
    act(() => latest().emitResult([['world', false]]))
    expect(textarea()).toHaveValue('hello world')
  })

  it('force-finishes after the stop grace period', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(micButton())
    const rec = latest()
    rec.stop.mockImplementation(() => {})
    fireEvent.click(micButton())
    act(() => vi.advanceTimersByTime(STOP_GRACE_MS))
    expect(rec.abort).toHaveBeenCalled()
    expect(micButton()).toHaveAttribute('aria-pressed', 'false')
  })
})
