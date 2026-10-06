import styles from './MessageBubble.module.css'

export default function MessageBubble({ role, text }) {
  const isUser = role === 'user'

  return (
    <div className={`${styles.row} ${isUser ? styles.user : ''}`}>
      {!isUser && (
        <div className={styles.avatar} aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v18M5 7h14M5 7l-3 7a3 3 0 0 0 6 0L5 7zm14 0l-3 7a3 3 0 0 0 6 0l-3-7zM8 21h8" />
          </svg>
        </div>
      )}
      <div className={styles.bubble}>{text}</div>
    </div>
  )
}
