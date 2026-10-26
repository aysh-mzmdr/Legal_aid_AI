import RichText from '../RichText/RichText.jsx'
import styles from './MessageBubble.module.css'

const STAGES = {
  searching: 'Searching the Constitution…',
  answering: 'Writing the answer…',
}

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g

// The model cites passages by retrieval rank ("[4] ... [1] ... [4]"). Display them as 1, 2, 3
// in order of first citation instead; a passage keeps its number every time it is cited.
function citationOrder(text, sources) {
  const order = new Map()
  for (const m of text.matchAll(CITATION)) {
    for (const n of m[1].split(',').map((x) => Number(x.trim()))) {
      if (!order.has(n) && sources.some((s) => s.n === n)) order.set(n, order.size + 1)
    }
  }
  return order
}

// message: { role, text, status?, sources?, cited?, followups?, error?, pending? }
// Older saved messages only have role and text.
export default function MessageBubble({ message, onOpenSource, onFollowup, showFollowups }) {
  const { role, text = '', status, sources: rawSources = [], cited = [], followups = [], error, pending } = message
  const isUser = role === 'user'
  const order = citationOrder(text, rawSources)
  const sources = rawSources.map((s) => ({ ...s, shown: order.get(s.n) }))
  const citedSources = cited.map((n) => sources.find((s) => s.n === n)).filter(Boolean)

  return (
    <div className={`${styles.row} ${isUser ? styles.user : ''}`}>
      {!isUser && (
        <div className={styles.avatar} aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v18M5 7h14M5 7l-3 7a3 3 0 0 0 6 0L5 7zm14 0l-3 7a3 3 0 0 0 6 0l-3-7zM8 21h8" />
          </svg>
        </div>
      )}

      <div className={styles.content}>
        <div className={styles.bubble} aria-live={pending ? 'polite' : undefined} aria-busy={pending || undefined}>
          {isUser ? (
            text
          ) : (
            <>
              {text && <RichText text={text} sources={sources} onCite={onOpenSource} />}
              {pending && !text && (
                <span className={styles.status}>
                  {STAGES[status] ?? 'Thinking…'}
                  <span className={styles.dots} aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </span>
                </span>
              )}
              {error && <p className={styles.error}>{error}</p>}
            </>
          )}
        </div>

        {citedSources.length > 0 && (
          <div className={styles.sources}>
            <span className={styles.sectionLabel}>Sources</span>
            <div className={styles.sourceList}>
              {citedSources.map((s) => (
                <button key={s.n} type="button" className={styles.source} onClick={() => onOpenSource(s)} title={s.title}>
                  <span className={styles.sourceNum}>{s.shown ?? s.n}</span>
                  {s.label}
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M7 17L17 7M9 7h8v8" />
                  </svg>
                </button>
              ))}
            </div>
          </div>
        )}

        {showFollowups && followups.length > 0 && (
          <div className={styles.followups}>
            <span className={styles.sectionLabel}>Ask next</span>
            {followups.map((q) => (
              <button key={q} type="button" className={styles.followup} onClick={() => onFollowup(q)}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
