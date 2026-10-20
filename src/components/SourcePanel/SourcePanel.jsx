import { useEffect, useRef, useState } from 'react'
import { fetchProvision } from '../../api/chat.js'
import styles from './SourcePanel.module.css'

// Splits provision text into plain text and "[Note: ...]" amendment notes (which may
// themselves contain brackets), so notes can be shown apart from the law itself.
function splitNotes(text) {
  const parts = []
  let i = 0
  while (i < text.length) {
    const start = text.indexOf('[Note: ', i)
    if (start === -1) {
      parts.push({ note: false, text: text.slice(i) })
      break
    }
    if (start > i) parts.push({ note: false, text: text.slice(i, start) })
    let depth = 0
    let end = start
    for (; end < text.length; end++) {
      if (text[end] === '[') depth++
      else if (text[end] === ']' && --depth === 0) break
    }
    parts.push({ note: true, text: text.slice(start + 7, end) })
    i = end + 1
  }
  return parts
}

// Shows the full text of a cited provision. Remount with key={source.id} to load another.
export default function SourcePanel({ source, onClose }) {
  const [provision, setProvision] = useState(null)
  const [error, setError] = useState(null)
  const closeRef = useRef(null)

  useEffect(() => {
    let live = true
    fetchProvision(source.id).then(
      (p) => live && setProvision(p),
      (e) => live && setError(e.message),
    )
    return () => {
      live = false
    }
  }, [source.id])

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const pages = provision?.pdf_pages ?? []

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <aside
        className={styles.panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.head}>
          <div className={styles.headText}>
            <span className={styles.badge}>{source.n}</span>
            <h2 id="source-title" className={styles.title}>
              {provision?.title ?? source.title ?? source.label}
            </h2>
          </div>
          <button ref={closeRef} type="button" className={styles.close} onClick={onClose} aria-label="Close source">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        {provision?.trail?.length > 0 && <p className={styles.trail}>{provision.trail.join(' › ')}</p>}

        <div className={styles.body}>
          {error && <p className={styles.muted}>{error}</p>}
          {!provision && !error && <p className={styles.muted}>Loading…</p>}
          {provision &&
            provision.text.split('\n').map((para, i) => (
              <p key={i} className={styles.para}>
                {splitNotes(para).map((part, j) =>
                  part.note ? (
                    <span key={j} className={styles.note}>
                      {part.text}
                    </span>
                  ) : (
                    part.text
                  ),
                )}
              </p>
            ))}
        </div>

        {provision && (
          <p className={styles.foot}>
            Constitution of India, {pages.length > 1 ? `PDF pages ${pages[0]}–${pages[pages.length - 1]}` : `PDF page ${pages[0]}`}
            . Notes in grey record amendments.
          </p>
        )}
      </aside>
    </div>
  )
}
