import { useRef, useState } from 'react'
import { Check, PanelLeftClose, PanelLeftOpen, Pencil, Plus, Trash2, X } from 'lucide-react'
import { groupSessionsByDate } from './sessionGroups.js'

function SessionItem({ session, active, onSelect, onRename, onDelete }) {
  const [mode, setMode] = useState('view') // view | rename | confirm-delete
  const [draft, setDraft] = useState(session.title)
  const editingRef = useRef(false) // blur after Enter/Escape must not commit again

  function startRename() {
    editingRef.current = true
    setDraft(session.title)
    setMode('rename')
  }

  function cancelRename() {
    editingRef.current = false
    setMode('view')
  }

  async function commitRename() {
    if (!editingRef.current) return
    editingRef.current = false
    const title = draft.trim()
    setMode('view')
    if (title && title !== session.title) await onRename(session.session_id, title)
  }

  if (mode === 'rename') {
    return (
      <div className="history-item active editing">
        <input
          className="history-rename-input"
          value={draft}
          maxLength={120}
          autoFocus
          aria-label="Chat title"
          onChange={e => setDraft(e.target.value)}
          onFocus={e => e.target.select()}
          onBlur={commitRename}
          onKeyDown={e => {
            if (e.key === 'Enter') { e.preventDefault(); commitRename() }
            if (e.key === 'Escape') cancelRename()
          }}
        />
      </div>
    )
  }

  return (
    <div className={`history-item ${active ? 'active' : ''}`}>
      <button
        type="button"
        className="history-item-content"
        onClick={() => onSelect(session.session_id)}
        aria-current={active ? 'true' : undefined}
      >
        <strong>{session.title}</strong>
        <p>{session.preview || 'Open to continue the conversation.'}</p>
      </button>
      {mode === 'confirm-delete' ? (
        <div className="history-item-actions visible">
          <button
            type="button"
            className="history-icon-btn danger"
            title="Confirm delete"
            aria-label={`Confirm delete ${session.title}`}
            onClick={() => { setMode('view'); onDelete(session.session_id) }}
          ><Check size={14} aria-hidden="true" /></button>
          <button
            type="button"
            className="history-icon-btn"
            title="Cancel"
            aria-label="Cancel delete"
            onClick={() => setMode('view')}
          ><X size={14} aria-hidden="true" /></button>
        </div>
      ) : (
        <div className="history-item-actions">
          <button
            type="button"
            className="history-icon-btn"
            title="Rename chat"
            aria-label={`Rename ${session.title}`}
            onClick={startRename}
          ><Pencil size={14} aria-hidden="true" /></button>
          <button
            type="button"
            className="history-icon-btn danger"
            title="Delete chat"
            aria-label={`Delete ${session.title}`}
            onClick={() => setMode('confirm-delete')}
          ><Trash2 size={14} aria-hidden="true" /></button>
        </div>
      )}
    </div>
  )
}

export default function Sidebar({
  sessions, activeId, collapsed: collapsedOnDesktop, mobileOpen,
  onToggleCollapse, onNewChat, onSelect, onRename, onDelete,
}) {
  // The mobile drawer always shows the full list.
  const collapsed = collapsedOnDesktop && !mobileOpen
  const groups = groupSessionsByDate(sessions)
  const classes = ['history-sidebar', collapsed && 'collapsed', mobileOpen && 'open'].filter(Boolean).join(' ')

  return (
    <aside className={classes} aria-label="Chat history">
      <div className="history-header">
        {!collapsed && <h2>Chats</h2>}
        <button
          type="button"
          className="icon-btn desktop-only"
          onClick={onToggleCollapse}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!collapsed}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <PanelLeftOpen size={18} aria-hidden="true" /> : <PanelLeftClose size={18} aria-hidden="true" />}
        </button>
      </div>

      <button
        className="new-chat-btn"
        onClick={onNewChat}
        type="button"
        aria-label="New chat"
        title="New chat"
      >
        <Plus size={16} aria-hidden="true" />
        {!collapsed && <span>New chat</span>}
      </button>

      {!collapsed && (
        <nav className="history-list" aria-label="Saved chats">
          {groups.length === 0 ? (
            <div className="empty-note">Your conversations will appear here.</div>
          ) : (
            groups.map(group => (
              <section key={group.label} className="history-group">
                <h3 className="history-group-label">{group.label}</h3>
                {group.sessions.map(session => (
                  <SessionItem
                    key={session.session_id}
                    session={session}
                    active={session.session_id === activeId}
                    onSelect={onSelect}
                    onRename={onRename}
                    onDelete={onDelete}
                  />
                ))}
              </section>
            ))
          )}
        </nav>
      )}
    </aside>
  )
}
