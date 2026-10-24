import { StrictMode } from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { TOAST_DEDUPE_MS, useToast } from '../useToast'
import App from '../App'

function Harness() {
  const { addToast, ToastContainer } = useToast(10_000)
  return (
    <div>
      <button onClick={() => addToast('Same message', 'error')}>Add</button>
      <button onClick={() => addToast('Same message', 'info')}>Add info</button>
      <ToastContainer />
    </div>
  )
}

describe('toast dedupe', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('shows identical toasts within 2s only once', () => {
    render(<Harness />)
    fireEvent.click(screen.getByText('Add'))
    fireEvent.click(screen.getByText('Add'))
    expect(screen.getAllByRole('alert')).toHaveLength(1)
  })

  it('shows the same message again after the dedupe window', () => {
    render(<Harness />)
    fireEvent.click(screen.getByText('Add'))
    act(() => vi.advanceTimersByTime(TOAST_DEDUPE_MS + 1))
    fireEvent.click(screen.getByText('Add'))
    expect(screen.getAllByRole('alert')).toHaveLength(2)
  })

  it('does not merge different types with the same text', () => {
    render(<Harness />)
    fireEvent.click(screen.getByText('Add'))
    fireEvent.click(screen.getByText('Add info'))
    expect(screen.getAllByRole('alert')).toHaveLength(2)
  })

  it('renders one toast per call under StrictMode', () => {
    render(<StrictMode><Harness /></StrictMode>)
    fireEvent.click(screen.getByText('Add'))
    expect(screen.getAllByRole('alert')).toHaveLength(1)
  })
})

describe('App error toast under StrictMode', () => {
  beforeEach(() => { window.HTMLElement.prototype.scrollIntoView = vi.fn() })
  afterEach(() => vi.unstubAllGlobals())

  it('a failed request produces exactly one toast', async () => {
    vi.stubGlobal('fetch', vi.fn(async url => {
      if (String(url).endsWith('/sessions')) {
        return new Response(JSON.stringify({ sessions: [] }), { status: 200 })
      }
      return new Response(JSON.stringify({ detail: 'Server exploded.' }), { status: 500 })
    }))
    render(<StrictMode><App /></StrictMode>)
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'hi' } })
    fireEvent.click(screen.getByLabelText('Send'))
    const toasts = await screen.findAllByRole('alert')
    expect(toasts).toHaveLength(1)
    expect(toasts[0]).toHaveTextContent('Server exploded.')
  })
})
