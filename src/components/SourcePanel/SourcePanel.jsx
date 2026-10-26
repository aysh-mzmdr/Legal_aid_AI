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
    parts.push({ note: true, text: text.slice(start + 7, end).trim() })
    i = end + 1
  }
  return parts
}

const CLAUSE = /^\((\d+[A-Z]?)\)\s*/ // (1), (2A)
const ROMAN = /^\(((?:x{0,3})(?:ix|iv|v?i{0,3}))\)\s*/ // (i), (iv), (xii)
const LETTER = /^\(([a-z]{1,2})\)\s*/ // (a), (aa)
const ENTRY = /^(\d+[A-Z]?)\.\s+/ // "1. Andhra Pradesh" in Schedule lists

// Marker and nesting level of one line: (1) is level 1, (a) level 2, (i) level 3.
// "(i)" right after "(h)" is the letter i, not the numeral.
function lineMarker(text, prev) {
  let m = text.match(CLAUSE)
  if (m) return { marker: `(${m[1]})`, level: 1, length: m[0].length }
  m = text.match(ROMAN)
  if (m && m[1] && prev?.marker !== '(h)') return { marker: `(${m[1]})`, level: 3, length: m[0].length }
  m = text.match(LETTER)
  if (m) return { marker: `(${m[1]})`, level: 2, length: m[0].length }
  m = text.match(ENTRY)
  if (m) return { marker: `${m[1]}.`, level: 1, length: m[0].length }
  return null
}

// Turns the provision text into lines of the law (with references to notes) plus a numbered
// list of the notes, so the law reads cleanly and amendments are listed below it.
// A note that opens a line, or stands on a line of its own, is referenced at the end of the
// line it introduces. A line without a marker (a proviso, say) is indented to the text of the
// clause above it.
function parseProvision(text) {
  const notes = []
  const lines = []
  let prev = null
  let pending = [] // note references waiting for the next line of text
  for (const raw of text.split('\n')) {
    const segments = []
    for (const part of splitNotes(raw)) {
      if (part.note) {
        notes.push(part.text)
        const ref = { note: notes.length }
        const last = segments[segments.length - 1]
        if (!segments.some((s) => s.text?.trim())) pending.push(ref)
        else {
          last.text = last.text.replace(/\s+$/, '')
          segments.push(ref)
        }
      } else if (part.text.trim() || segments.length) {
        segments.push({ text: part.text })
      }
    }
    const first = segments.find((s) => s.text?.trim())
    if (!first) continue
    first.text = first.text.trimStart()
    const found = lineMarker(first.text, prev)
    if (found) {
      first.text = first.text.slice(found.length)
      prev = found
    }
    segments.push(...pending)
    pending = []
    lines.push({ marker: found?.marker, indent: found ? found.level - 1 : (prev?.level ?? 0), segments })
  }
  if (pending.length && lines.length) lines[lines.length - 1].segments.push(...pending)
  return { lines, notes }
}

// "Article 22: Protection against arrest ... (part 1 of 3, clauses (1)–(3))" ->
// title "Protection against arrest ...", part "Part 1 of 3".
function splitTitle(title = '') {
  const partMatch = title.match(/\s*\(part (\d+) of (\d+)[^)]*(?:\([^)]*\)[^)]*)*\)\s*$/)
  const base = partMatch ? title.slice(0, partMatch.index) : title
  const colon = base.indexOf(': ')
  return {
    title: colon === -1 ? base : base.slice(colon + 2),
    part: partMatch ? `Part ${partMatch[1]} of ${partMatch[2]}` : null,
  }
}

// "... (with effect from 3-1-1977)." -> text without it, date "3-1-1977".
function splitNoteDate(note) {
  const m = note.match(/\s*\((?:with effect from|w\.e\.f\.)\s*([^)]+)\)\.?\s*$/i)
  return m ? { text: note.slice(0, m.index).replace(/[,\s]+$/, '') + '.', date: m[1].trim() } : { text: note, date: null }
}

function Skeleton() {
  return (
    <div className={styles.skeleton} aria-label="Loading">
      {[92, 100, 84, 96, 70].map((w, i) => (
        <span key={i} style={{ width: `${w}%` }} />
      ))}
    </div>
  )
}

// Shows the full text of a cited provision. Remount with key={source.id} to load another.
export default function SourcePanel({ source, onClose }) {
  const [provision, setProvision] = useState(null)
  const [error, setError] = useState(null)
  const [activeNote, setActiveNote] = useState(null)
  const closeRef = useRef(null)
  const noteRefs = useRef({})

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

  const heading = splitTitle(provision?.title ?? source.title ?? source.label)
  const label = provision?.label ?? source.label
  const pages = provision?.pdf_pages ?? []
  const { lines, notes } = provision ? parseProvision(provision.text) : { lines: [], notes: [] }
  const hasLaw = lines.length > 0

  const showNote = (k) => {
    setActiveNote(k)
    noteRefs.current[k]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

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
          <span className={styles.badge} aria-label={`Source ${source.shown ?? source.n}`}>
            {source.shown ?? source.n}
          </span>
          <div className={styles.headText}>
            <p className={styles.eyebrow}>
              {label}
              {heading.part && <span className={styles.part}>{heading.part}</span>}
            </p>
            <h2 id="source-title" className={styles.title}>
              {heading.title}
            </h2>
          </div>
          <button ref={closeRef} type="button" className={styles.close} onClick={onClose} aria-label="Close source">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        {provision?.trail?.length > 0 && (
          <nav className={styles.trail} aria-label="Location in the Constitution">
            {provision.trail.map((t, i) => (
              <span key={i}>{t}</span>
            ))}
          </nav>
        )}

        <div className={styles.scroll}>
          <section className={styles.law}>
            {error && <p className={styles.muted}>{error}</p>}
            {!provision && !error && <Skeleton />}
            {provision && !hasLaw && (
              <p className={styles.muted}>This part has no text of its own; it is recorded in the amendment note below.</p>
            )}
            {lines.map((line, i) => (
              <div
                key={i}
                className={`${styles.line} ${line.marker ? '' : styles.plain}`}
                style={{ '--level': line.indent }}
              >
                {line.marker && <span className={styles.marker}>{line.marker}</span>}
                <p className={styles.lineText}>
                  {line.segments.map((s, j) =>
                    s.note ? (
                      <button
                        key={j}
                        type="button"
                        className={styles.noteRef}
                        onClick={() => showNote(s.note)}
                        aria-label={`Amendment note ${s.note}`}
                      >
                        {s.note}
                      </button>
                    ) : (
                      <span key={j}>{s.text}</span>
                    ),
                  )}
                </p>
              </div>
            ))}
          </section>

          {notes.length > 0 && (
            <section className={styles.notes} aria-labelledby="notes-title">
              <h3 id="notes-title" className={styles.notesTitle}>
                Amendment notes
              </h3>
              <ol className={styles.noteList}>
                {notes.map((note, i) => {
                  const k = i + 1
                  const { text, date } = splitNoteDate(note)
                  return (
                    <li
                      key={k}
                      ref={(el) => (noteRefs.current[k] = el)}
                      className={`${styles.note} ${activeNote === k ? styles.noteActive : ''}`}
                    >
                      <span className={styles.noteNum}>{k}</span>
                      <div>
                        <p className={styles.noteText}>{text}</p>
                        {date && <p className={styles.noteDate}>In force from {date}</p>}
                      </div>
                    </li>
                  )
                })}
              </ol>
            </section>
          )}
        </div>

        {provision && pages.length > 0 && (
          <p className={styles.foot}>
            Constitution of India · {pages.length > 1 ? `PDF pages ${pages[0]}–${pages[pages.length - 1]}` : `PDF page ${pages[0]}`}
          </p>
        )}
      </aside>
    </div>
  )
}
