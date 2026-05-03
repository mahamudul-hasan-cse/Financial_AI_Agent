function SessionsSidebar({ sessions, activeSessionId, onCreateNew, onSelect, onRename }) {
  return (
    <aside className="history-sidebar">
      <div className="history-header">
        <p className="eyebrow">Research</p>
        <h2>Saved Sessions</h2>
      </div>

      <button className="new-chat-btn" onClick={onCreateNew} type="button">
        + New Session
      </button>

      <div className="history-list">
        {sessions.length === 0 ? (
          <div className="empty-note">Saved research will appear here after your first analysis.</div>
        ) : (
          sessions.map(session => (
            <div
              key={session.session_id}
              className={`history-item ${activeSessionId === session.session_id ? 'active' : ''}`}
            >
              <button
                type="button"
                className="history-item-content"
                onClick={() => onSelect(session.session_id)}
              >
                <strong>{session.title}</strong>
                <p>{session.preview || 'No preview yet'}</p>
              </button>
              <small style={{ fontSize: 10, color: 'var(--text-dim)', flexShrink: 0, paddingTop: 2 }}>
                {new Date(session.updated_at * 1000).toLocaleDateString([], { month: 'short', day: 'numeric' })}
              </small>
            </div>
          ))
        )}
      </div>

      {activeSessionId && (
        <div style={{ padding: '10px 16px 16px' }}>
          <button
            className="ghost-btn"
            style={{ width: '100%', fontSize: 12 }}
            onClick={onRename}
            type="button"
          >
            Rename Session
          </button>
        </div>
      )}
    </aside>
  )
}

export default SessionsSidebar
