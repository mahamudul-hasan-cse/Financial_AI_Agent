import { useState, useCallback, useEffect, useRef, useSyncExternalStore } from 'react'
import { CircleAlert, CircleCheck, Info } from 'lucide-react'

let _idCounter = 0

// Identical toasts (same type + message) within this window are shown once.
export const TOAST_DEDUPE_MS = 2000

const ICONS = { success: CircleCheck, error: CircleAlert, info: Info }

// Tiny per-hook store, so <ToastContainer> can be a stable component that
// subscribes on its own. (A component *defined* inside the hook would be a
// new type on every render, remounting the toasts each time.)
function createToastStore() {
  let toasts = []
  const listeners = new Set()
  return {
    get: () => toasts,
    set(next) {
      toasts = typeof next === 'function' ? next(toasts) : next
      listeners.forEach(fn => fn())
    },
    subscribe(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
  }
}

function makeContainer(store, removeToast, duration) {
  return function ToastContainer() {
    const toasts = useSyncExternalStore(store.subscribe, store.get, store.get)
    if (toasts.length === 0) return null
    return (
      <div className="toast-container" role="region" aria-label="Notifications">
        {toasts.map(t => {
          const Icon = ICONS[t.type] || ICONS.info
          return (
            <div
              key={t.id}
              className={`toast ${t.type}`}
              style={{ '--toast-duration': `${duration}ms` }}
              onClick={() => removeToast(t.id)}
              role="alert"
            >
              <Icon className="toast-icon" size={16} aria-hidden="true" />
              <span>{t.message}</span>
            </div>
          )
        })}
      </div>
    )
  }
}

/**
 * Simple toast notification hook.
 *
 * Usage:
 *   const { addToast, ToastContainer } = useToast()
 *   addToast('File downloaded', 'success')
 *
 * Returns a ToastContainer component to render in your JSX.
 */
export function useToast(defaultDuration = 3500) {
  const [store] = useState(createToastStore)
  const toasts = useSyncExternalStore(store.subscribe, store.get, store.get)
  const timersRef = useRef({})
  const recentRef = useRef(new Map()) // "type:message" -> { id, at }

  const removeToast = useCallback((id) => {
    store.set(prev => prev.filter(t => t.id !== id))
    clearTimeout(timersRef.current[id])
    delete timersRef.current[id]
  }, [store])

  const addToast = useCallback((message, type = 'info', duration = defaultDuration) => {
    const key = `${type}:${message}`
    const now = Date.now()
    const recent = recentRef.current.get(key)
    if (recent && now - recent.at < TOAST_DEDUPE_MS) return recent.id
    const id = ++_idCounter
    recentRef.current.set(key, { id, at: now })
    store.set(prev => [...prev, { id, message, type }])
    timersRef.current[id] = setTimeout(() => removeToast(id), duration)
    return id
  }, [defaultDuration, removeToast, store])

  useEffect(() => {
    const timers = timersRef.current
    return () => Object.values(timers).forEach(clearTimeout)
  }, [])

  const [ToastContainer] = useState(() => makeContainer(store, removeToast, defaultDuration))

  return { toasts, addToast, removeToast, ToastContainer }
}
