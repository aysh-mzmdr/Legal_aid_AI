import { useLayoutEffect, useRef, useState } from 'react'
import styles from './ChatInput.module.css'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const [multiline, setMultiline] = useState(false)
  const inputRef = useRef(null)

  // Grow the textarea with its content (up to the CSS max-height) instead of scrolling.
  useLayoutEffect(() => {
    const el = inputRef.current
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
    setMultiline(el.scrollHeight > parseFloat(getComputedStyle(el).lineHeight) * 1.5)
  }, [value])

  const submit = () => {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className={`${styles.wrap} ${multiline ? styles.multiline : ''}`}>
      <textarea
        ref={inputRef}
        className={styles.input}
        rows={1}
        value={value}
        placeholder="Having a query?"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        aria-label="Message"
      />
      <button
        type="button"
        className={styles.send}
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="Send message"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 19V5M5 12l7-7 7 7" />
        </svg>
      </button>
    </div>
  )
}
