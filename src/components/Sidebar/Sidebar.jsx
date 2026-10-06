import styles from './Sidebar.module.css'

export default function Sidebar({ conversations, activeId, onSelect, onNew, onDelete, open, onClose }) {
  return (
    <>
      {open && <div className={styles.backdrop} onClick={onClose} />}
      <aside className={`${styles.sidebar} ${open ? styles.open : ''}`}>
        <span className={styles.brand}>
          Legal <span className={styles.accentWord}>Aid</span> AI
        </span>

        <button type="button" className={styles.newChat} onClick={onNew}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
          New chat
        </button>

        <p className={styles.label}>History</p>

        <ul className={styles.list}>
          {conversations.map((c) => (
            <li key={c.id} className={styles.item}>
              <button
                type="button"
                className={`${styles.select} ${c.id === activeId ? styles.active : ''}`}
                onClick={() => onSelect(c.id)}
              >
                {c.title}
              </button>
              <button
                type="button"
                className={styles.delete}
                onClick={() => onDelete(c.id)}
                aria-label={`Delete ${c.title}`}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </>
  )
}
