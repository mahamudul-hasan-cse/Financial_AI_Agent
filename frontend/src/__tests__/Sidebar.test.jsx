import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import App from '../App'
import { groupSessionsByDate } from '../sessionGroups'

const NOW = new Date(2026, 9, 24, 15, 0, 0)
const sec = d => d.getTime() / 1000

describe('groupSessionsByDate', () => {
  it('buckets by local calendar day', () => {
    const sessions = [
      { session_id: 'a', updated_at: sec(new Date(2026, 9, 24, 0, 5)) },
      { session_id: 'b', updated_at: sec(new Date(2026, 9, 23, 23, 59)) },
      { session_id: 'c', updated_at: sec(new Date(2026, 9, 22, 12, 0)) },
    ]
    expect(groupSessionsByDate(sessions, NOW).map(g => [g.label, g.sessions.map(s => s.session_id)]))
      .toEqual([['Today', ['a']], ['Yesterday', ['b']], ['Earlier', ['c']]])
  })

  it('omits empty groups', () => {
    expect(groupSessionsByDate([], NOW)).toEqual([])
  })
})

describe('App sidebar', () => {
  let fetchMock
  const now = Date.now() / 1000
  const sessions = () => [
    { session_id: 'sess_today', title: 'Apple deep dive', preview: 'AAPL?', updated_at: now },
    { session_id: 'sess_old', title: 'Old chat', preview: 'old', updated_at: now - 10 * 86400 },
  ]

  beforeEach(() => {
    localStorage.clear()
    window.HTMLElement.prototype.scrollIntoView = vi.fn()
    fetchMock = vi.fn(async (url, opts = {}) => {
      const path = String(url)
      if (path.endsWith('/sessions')) return new Response(JSON.stringify({ sessions: sessions() }), { status: 200 })
      if (opts.method === 'PATCH') {
        const { title } = JSON.parse(opts.body)
        return new Response(JSON.stringify({ ...sessions()[0], title }), { status: 200 })
      }
      if (opts.method === 'DELETE') return new Response(JSON.stringify({ status: 'cleared' }), { status: 200 })
      return new Response('{}', { status: 404 })
    })
    vi.stubGlobal('fetch', fetchMock)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('groups saved chats under date headings', async () => {
    render(<App />)
    await screen.findByText('Apple deep dive')
    expect(screen.getByRole('heading', { name: 'Today' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Earlier' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Yesterday' })).not.toBeInTheDocument()
  })

  it('renames a chat with PATCH /sessions/{id}', async () => {
    render(<App />)
    await screen.findByText('Apple deep dive')
    fireEvent.click(screen.getByLabelText('Rename Apple deep dive'))
    const input = screen.getByLabelText('Chat title')
    fireEvent.change(input, { target: { value: 'AAPL notes' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(await screen.findByText('AAPL notes')).toBeInTheDocument()
    const patches = fetchMock.mock.calls.filter(([, o]) => o?.method === 'PATCH')
    expect(patches).toHaveLength(1)
    expect(String(patches[0][0])).toMatch(/\/sessions\/sess_today$/)
    expect(JSON.parse(patches[0][1].body)).toEqual({ title: 'AAPL notes' })
  })

  it('escape cancels a rename without a request', async () => {
    render(<App />)
    await screen.findByText('Apple deep dive')
    fireEvent.click(screen.getByLabelText('Rename Apple deep dive'))
    const input = screen.getByLabelText('Chat title')
    fireEvent.change(input, { target: { value: 'nope' } })
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(screen.getByText('Apple deep dive')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([, o]) => o?.method === 'PATCH')).toBe(false)
  })

  it('deletes a chat after confirmation', async () => {
    render(<App />)
    await screen.findByText('Old chat')
    fireEvent.click(screen.getByLabelText('Delete Old chat'))
    expect(fetchMock.mock.calls.some(([, o]) => o?.method === 'DELETE')).toBe(false)
    fireEvent.click(screen.getByLabelText('Confirm delete Old chat'))

    await waitFor(() => expect(screen.queryByText('Old chat')).not.toBeInTheDocument())
    const del = fetchMock.mock.calls.find(([, o]) => o?.method === 'DELETE')
    expect(String(del[0])).toMatch(/\/chat\/sess_old$/)
  })

  it('collapses on desktop and remembers it', async () => {
    const { unmount } = render(<App />)
    await screen.findByText('Apple deep dive')
    fireEvent.click(screen.getByLabelText('Collapse sidebar'))
    expect(screen.queryByText('Apple deep dive')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'New chat' })).toBeInTheDocument()
    expect(localStorage.getItem('sidebar-collapsed')).toBe('1')
    unmount()

    render(<App />)
    const aside = screen.getByRole('complementary', { name: 'Chat history' })
    expect(aside).toHaveClass('collapsed')
    fireEvent.click(within(aside).getByLabelText('Expand sidebar'))
    expect(await screen.findByText('Apple deep dive')).toBeInTheDocument()
  })
})
