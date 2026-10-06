import { useEffect, useRef, useState } from 'react'
import MessageBubble from '../MessageBubble/MessageBubble.jsx'
import ChatInput from '../ChatInput/ChatInput.jsx'
import Sidebar from '../Sidebar/Sidebar.jsx'
import styles from './ChatPage.module.css'

const SUGGESTIONS = [
  'What are my fundamental rights?',
  'How do I file an FIR?',
  'Explain Article 21',
]

const GREETING = {
  role: 'assistant',
  text: 'Hello! I can help you understand your legal rights and the Constitution of India. What would you like to know?',
}

const isMobile = () => window.matchMedia('(max-width: 800px)').matches

const STORAGE_KEY = 'legal-aid-ai:conversations'

const newConversation = () => ({
  id: crypto.randomUUID(),
  title: 'New chat',
  messages: [GREETING],
})

const loadConversations = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY))
    if (Array.isArray(saved) && saved.length) return saved
  } catch {
    // ignore corrupt or unavailable storage
  }
  return [newConversation()]
}

export default function ChatPage() {
  const [conversations, setConversations] = useState(loadConversations)
  const [activeId, setActiveId] = useState(() => conversations[0].id)
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(() => !isMobile())
  const endRef = useRef(null)

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0]
  const messages = active.messages

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
    } catch {
      // storage full or unavailable
    }
  }, [conversations])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, activeId])

  const appendMessage = (id, message, title) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, title: title ?? c.title, messages: [...c.messages, message] } : c,
      ),
    )
  }

  const handleSend = (text) => {
    const id = active.id
    const title = active.messages.length === 1 ? text.slice(0, 40) : undefined
    appendMessage(id, { role: 'user', text }, title)
    setLoading(true)

    // TODO: replace with a real call to the AI backend.
    setTimeout(() => {
      appendMessage(id, {
        role: 'assistant',
        text: 'The AI backend is not connected yet. Your question has been received.',
      })
      setLoading(false)
    }, 900)
  }

  const handleNew = () => {
    // Reuse an existing empty chat instead of piling up blanks.
    const empty = conversations.find((c) => c.messages.length === 1)
    if (empty) {
      setActiveId(empty.id)
    } else {
      const conv = newConversation()
      setConversations((prev) => [conv, ...prev])
      setActiveId(conv.id)
    }
    if (isMobile()) setSidebarOpen(false)
  }

  const handleSelect = (id) => {
    setActiveId(id)
    if (isMobile()) setSidebarOpen(false)
  }

  const handleDelete = (id) => {
    const remaining = conversations.filter((c) => c.id !== id)
    if (!remaining.length) {
      const conv = newConversation()
      setConversations([conv])
      setActiveId(conv.id)
      return
    }
    setConversations(remaining)
    if (id === active.id) setActiveId(remaining[0].id)
  }

  const showSuggestions = messages.length === 1

  return (
    <div className={styles.page}>
      <div className={styles.body}>
        <Sidebar
          conversations={conversations}
          activeId={active.id}
          onSelect={handleSelect}
          onNew={handleNew}
          onDelete={handleDelete}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        <main className={styles.panel}>
          <button
            type="button"
            className={styles.menu}
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle chat history"
            aria-expanded={sidebarOpen}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M4 7h16M4 12h16M4 17h16" />
            </svg>
          </button>
          <div className={styles.glow} />

          <div className={styles.scroll}>
            <div className={styles.thread}>
              {showSuggestions && (
                <div className={styles.hero}>
                  <h1 className={styles.title}>
                    Ask your <span className={styles.accentWord}>legal</span> question
                  </h1>
                  <p className={styles.subtitle}>Clear, calm guidance on your rights, in plain language.</p>
                </div>
              )}

              {messages.map((m, i) => (
                <MessageBubble key={i} role={m.role} text={m.text} />
              ))}
              {loading && <MessageBubble role="assistant" text="Thinking..." />}
              <div ref={endRef} />
            </div>
          </div>

          <div className={styles.footer}>
            {showSuggestions && (
              <div className={styles.suggestions}>
                {SUGGESTIONS.map((s) => (
                  <button key={s} type="button" className={styles.chip} onClick={() => handleSend(s)} disabled={loading}>
                    {s}
                  </button>
                ))}
              </div>
            )}
            <ChatInput onSend={handleSend} disabled={loading} />
            <p className={styles.disclaimer}>General information only, not a substitute for a licensed attorney.</p>
          </div>
        </main>
      </div>
    </div>
  )
}
