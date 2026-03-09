import { useState, useCallback, useRef } from 'react'

let _idCounter = 0

/**
 * Simple toast notification hook.
 *
 * Usage:
 *   const { toasts, addToast, ToastContainer } = useToast()
 *   addToast('File downloaded', 'success')
 *   addToast('Something went wrong', 'error')
 *
 * Returns a ToastContainer component to render in your JSX.
 */
export function useToast(defaultDuration = 3500) {
  const [toasts, setToasts] = useState([])
  const timersRef = useRef({})

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
    clearTimeout(timersRef.current[id])
    delete timersRef.current[id]
  }, [])

  const addToast = useCallback((message, type = 'info', duration = defaultDuration) => {
    const id = ++_idCounter
    setToasts(prev => [...prev, { id, message, type }])
    timersRef.current[id] = setTimeout(() => removeToast(id), duration)
    return id
  }, [defaultDuration, removeToast])

  function ToastContainer() {
    if (toasts.length === 0) return null
    const icons = { success: '\u2713', error: '\u2717', info: '\u24D8' }
    return (
      <div className="toast-container">
        {toasts.map(t => (
          <div
            key={t.id}
            className={`toast ${t.type}`}
            style={{ '--toast-duration': `${defaultDuration}ms` }}
            onClick={() => removeToast(t.id)}
            role="alert"
          >
            <span className="toast-icon">{icons[t.type] || icons.info}</span>
            {t.message}
          </div>
        ))}
      </div>
    )
  }

  return { toasts, addToast, removeToast, ToastContainer }
}
