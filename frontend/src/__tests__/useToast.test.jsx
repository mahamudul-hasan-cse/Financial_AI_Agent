import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import { useToast } from '../useToast'

function ToastTestHarness() {
  const { addToast, ToastContainer } = useToast(3500)
  return (
    <div>
      <button onClick={() => addToast('Success!', 'success')}>Add Success</button>
      <button onClick={() => addToast('Error!', 'error')}>Add Error</button>
      <button onClick={() => addToast('Info!', 'info')}>Add Info</button>
      <ToastContainer />
    </div>
  )
}

describe('useToast', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('shows no toasts initially', () => {
    render(<ToastTestHarness />)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('displays a success toast when triggered', () => {
    render(<ToastTestHarness />)
    fireEvent.click(screen.getByText('Add Success'))
    expect(screen.getByText('Success!')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveClass('success')
  })

  it('displays an error toast', () => {
    render(<ToastTestHarness />)
    fireEvent.click(screen.getByText('Add Error'))
    expect(screen.getByText('Error!')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveClass('error')
  })

  it('auto-dismisses after duration', () => {
    render(<ToastTestHarness />)
    fireEvent.click(screen.getByText('Add Info'))
    expect(screen.getByText('Info!')).toBeInTheDocument()
    act(() => { vi.advanceTimersByTime(4000) })
    expect(screen.queryByText('Info!')).not.toBeInTheDocument()
  })

  it('dismisses toast on click', () => {
    render(<ToastTestHarness />)
    fireEvent.click(screen.getByText('Add Success'))
    fireEvent.click(screen.getByRole('alert'))
    expect(screen.queryByText('Success!')).not.toBeInTheDocument()
  })

  it('can display multiple toasts', () => {
    render(<ToastTestHarness />)
    fireEvent.click(screen.getByText('Add Success'))
    fireEvent.click(screen.getByText('Add Error'))
    expect(screen.getByText('Success!')).toBeInTheDocument()
    expect(screen.getByText('Error!')).toBeInTheDocument()
    expect(screen.getAllByRole('alert')).toHaveLength(2)
  })
})
