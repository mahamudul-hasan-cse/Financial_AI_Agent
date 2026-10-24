import { useCallback, useEffect, useRef, useState } from 'react'

export const SILENCE_TIMEOUT_MS = 10_000
export const MAX_LISTEN_MS = 60_000
// After stop(), give the browser this long to deliver onend before giving up.
export const STOP_GRACE_MS = 2_000

export const VOICE_LANGUAGES = [
  { code: 'en-US', label: 'English (US)', short: 'EN' },
  { code: 'bn-BD', label: 'বাংলা (Bangladesh)', short: 'BN' },
]

export function getSpeechRecognitionConstructor() {
  if (typeof window === 'undefined') return null
  return window.SpeechRecognition || window.webkitSpeechRecognition || null
}

export function joinTranscript(...parts) {
  return parts.map(p => (p || '').trim()).filter(Boolean).join(' ')
}

/**
 * Browser speech recognition with the guard rails the composer needs.
 *
 * - start() is ignored while a session is active (double-start guard); a
 *   start while the previous session is still winding down aborts it.
 * - stop() is graceful; stopping again (or no onend within STOP_GRACE_MS)
 *   force-finishes, so a browser that never fires onend can't wedge it.
 * - Stops itself after `silenceMs` without a result (reset on every
 *   result) and after `maxMs` in total.
 * - Aborts on unmount.
 *
 * Callbacks (latest version always used):
 *   onTranscript(text)   live text: base + final + interim results
 *   onFinish(text, info) once per session; info = { error, spoke }
 *   onError(code)        recognition error (never for 'aborted')
 */
export function useSpeechInput({
  lang = 'en-US',
  silenceMs = SILENCE_TIMEOUT_MS,
  maxMs = MAX_LISTEN_MS,
  onTranscript,
  onFinish,
  onError,
} = {}) {
  const supported = !!getSpeechRecognitionConstructor()
  const [listening, setListening] = useState(false)
  const sessionRef = useRef(null)
  const callbacksRef = useRef({})

  useEffect(() => {
    callbacksRef.current = { onTranscript, onFinish, onError }
  })

  const clearTimers = session => {
    clearTimeout(session.silenceTimer)
    clearTimeout(session.maxTimer)
    clearTimeout(session.graceTimer)
  }

  const finish = useCallback(session => {
    if (session.done) return
    session.done = true
    clearTimers(session)
    if (sessionRef.current === session) {
      sessionRef.current = null
      setListening(false)
    }
    const text = joinTranscript(session.base, session.finalText, session.interim)
    callbacksRef.current.onFinish?.(text, {
      error: session.error,
      spoke: !!(session.finalText || session.interim),
    })
  }, [])

  // Finish now (keeping the text) and cut the engine off.
  const forceFinish = useCallback(session => {
    finish(session)
    try { session.rec.abort() } catch { /* already ended */ }
  }, [finish])

  // Graceful stop; a second stop, or no onend within STOP_GRACE_MS, forces it.
  const stopSession = useCallback(session => {
    if (session.done) return
    if (session.stopping) { forceFinish(session); return }
    session.stopping = true
    clearTimers(session)
    session.graceTimer = setTimeout(() => forceFinish(session), STOP_GRACE_MS)
    try { session.rec.stop() } catch { forceFinish(session) }
  }, [forceFinish])

  const armSilence = useCallback(session => {
    clearTimeout(session.silenceTimer)
    session.silenceTimer = setTimeout(() => stopSession(session), silenceMs)
  }, [silenceMs, stopSession])

  // Drop the session without calling onFinish (new chat, unmount).
  const abort = useCallback(() => {
    const session = sessionRef.current
    if (!session) return
    session.done = true
    clearTimers(session)
    sessionRef.current = null
    setListening(false)
    try { session.rec.abort() } catch { /* already ended */ }
  }, [])

  const start = useCallback((base = '') => {
    const Ctor = getSpeechRecognitionConstructor()
    if (!Ctor) return false
    const previous = sessionRef.current
    if (previous && !previous.stopping) return false
    // Still winding down: keep its text, then cut it off.
    if (previous) forceFinish(previous)

    const rec = new Ctor()
    rec.lang = lang
    rec.interimResults = true
    rec.continuous = true
    rec.maxAlternatives = 1
    const session = {
      rec, base, finalText: '', interim: '', error: null,
      stopping: false, done: false, silenceTimer: null, maxTimer: null, graceTimer: null,
    }

    rec.onresult = e => {
      if (session.done) return
      // e.results holds every result of this session, so rebuild from it
      // (appending would duplicate finals that a later event repeats).
      let finalText = ''
      let interim = ''
      for (const result of Array.from(e.results)) {
        const text = result[0]?.transcript || ''
        if (result.isFinal) finalText = joinTranscript(finalText, text)
        else interim = joinTranscript(interim, text)
      }
      session.finalText = finalText
      session.interim = interim
      if (!session.stopping) armSilence(session)
      callbacksRef.current.onTranscript?.(
        joinTranscript(session.base, session.finalText, session.interim),
      )
    }
    rec.onerror = e => {
      if (session.done) return
      session.error = e?.error || 'unknown'
      if (session.error !== 'aborted') callbacksRef.current.onError?.(session.error)
    }
    rec.onend = () => finish(session)

    sessionRef.current = session
    setListening(true)
    armSilence(session)
    session.maxTimer = setTimeout(() => stopSession(session), maxMs)
    try {
      rec.start()
    } catch {
      session.done = true
      clearTimers(session)
      sessionRef.current = null
      setListening(false)
      callbacksRef.current.onError?.('start-failed')
      return false
    }
    return true
  }, [lang, maxMs, armSilence, finish, forceFinish, stopSession])

  const stop = useCallback(() => {
    if (sessionRef.current) stopSession(sessionRef.current)
  }, [stopSession])

  useEffect(() => abort, [abort])

  return { supported, listening, start, stop, abort }
}
