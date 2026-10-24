import { describe, it, expect, beforeEach } from 'vitest'
import { createSSEParser, getClientId, interpretEvent } from '../api'

describe('createSSEParser', () => {
  it('buffers events split across network chunks', () => {
    const push = createSSEParser()
    expect(push('data: "Hel')).toEqual([])
    expect(push('lo"\n')).toEqual([])
    expect(push('\ndata: "Wor')).toEqual(['"Hello"'])
    expect(push('ld"\n\ndata: [DONE]\n\n')).toEqual(['"World"', '[DONE]'])
  })

  it('handles several events in one chunk and CRLF framing', () => {
    const push = createSSEParser()
    expect(push('data: a\r\n\r\ndata: b\n\n')).toEqual(['a', 'b'])
  })

  it('flushes a trailing event without blank line at end of stream', () => {
    const push = createSSEParser()
    expect(push('data: [DONE]')).toEqual([])
    expect(push('', true)).toEqual(['[DONE]'])
  })

  it('ignores comments and joins multi-line data', () => {
    const push = createSSEParser()
    expect(push(': keep-alive\n\ndata: one\ndata: two\n\n')).toEqual(['one\ntwo'])
  })
})

describe('interpretEvent', () => {
  it('classifies protocol events', () => {
    expect(interpretEvent('[DONE]')).toEqual({ type: 'done' })
    expect(interpretEvent('[NLP_META]{"intent":"news"}')).toEqual({ type: 'nlp', value: { intent: 'news' } })
    expect(interpretEvent('[FINAL]"final text"')).toEqual({ type: 'final', value: 'final text' })
    expect(interpretEvent('[RESEARCH_DATA]{"a":1}')).toEqual({ type: 'research', value: { a: 1 } })
    expect(interpretEvent('[ERROR] boom')).toEqual({ type: 'error', value: 'boom' })
    expect(interpretEvent('"line\\nbreak"')).toEqual({ type: 'text', value: 'line\nbreak' })
  })

  it('parses retry status events', () => {
    expect(interpretEvent('[STATUS]{"type":"status","message":"Retrying in 4s…"}'))
      .toEqual({ type: 'status', value: 'Retrying in 4s…' })
    expect(interpretEvent('[STATUS]{oops')).toEqual({ type: 'ignored' })
  })

  it('never appends unparseable fragments as text', () => {
    expect(interpretEvent('"partial')).toEqual({ type: 'ignored' })
    expect(interpretEvent('[NLP_META]{broken')).toEqual({ type: 'ignored' })
  })
})

describe('getClientId', () => {
  beforeEach(() => localStorage.clear())

  it('creates a valid id once and reuses it', () => {
    const id = getClientId()
    expect(id).toMatch(/^[A-Za-z0-9_-]{16,128}$/)
    expect(getClientId()).toBe(id)
  })

  it('replaces an invalid stored id', () => {
    localStorage.setItem('financial-agent-client-id', 'bad id!')
    expect(getClientId()).toMatch(/^c_[0-9a-f]{32}$/)
  })
})
