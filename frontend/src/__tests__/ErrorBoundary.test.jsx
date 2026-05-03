import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import ErrorBoundary, { MessageErrorBoundary } from '../ErrorBoundary'

// Suppress console.error noise from React error boundaries during tests
const originalError = console.error
beforeEach(() => {
  console.error = vi.fn()
})
afterEach(() => {
  console.error = originalError
})

function ThrowingComponent({ message = 'Test error' }) {
  throw new Error(message)
}

function GoodComponent() {
  return <p>Everything is fine</p>
}

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <GoodComponent />
      </ErrorBoundary>
    )
    expect(screen.getByText('Everything is fine')).toBeInTheDocument()
  })

  it('renders fallback UI when child throws', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText(/unexpected error/)).toBeInTheDocument()
    expect(screen.getByText('Reload Page')).toBeInTheDocument()
  })

  it('does not affect siblings outside the boundary', () => {
    render(
      <div>
        <p>Outside</p>
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      </div>
    )
    expect(screen.getByText('Outside')).toBeInTheDocument()
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })
})

describe('MessageErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <MessageErrorBoundary>
        <GoodComponent />
      </MessageErrorBoundary>
    )
    expect(screen.getByText('Everything is fine')).toBeInTheDocument()
  })

  it('renders inline error message when child throws', () => {
    render(
      <MessageErrorBoundary>
        <ThrowingComponent />
      </MessageErrorBoundary>
    )
    expect(screen.getByText('Failed to render this message.')).toBeInTheDocument()
  })

  it('isolates errors per message boundary', () => {
    render(
      <div>
        <MessageErrorBoundary>
          <GoodComponent />
        </MessageErrorBoundary>
        <MessageErrorBoundary>
          <ThrowingComponent />
        </MessageErrorBoundary>
      </div>
    )
    expect(screen.getByText('Everything is fine')).toBeInTheDocument()
    expect(screen.getByText('Failed to render this message.')).toBeInTheDocument()
  })
})
