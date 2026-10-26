import { useCallback, useEffect, useRef, useState } from 'react'
import { streamChat } from '../../api/chat.js'
import MessageBubble from '../MessageBubble/MessageBubble.jsx'
import ChatInput from '../ChatInput/ChatInput.jsx'
import Sidebar from '../Sidebar/Sidebar.jsx'
import SourcePanel from '../SourcePanel/SourcePanel.jsx'
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

// An answer still streaming when the page was closed can never finish: mark it.
const settle = (m) =>
  m.pending ? { ...m, pending: false, error: m.text ? m.error : 'This answer was interrupted.' } : m

const HISTORY_TURNS = 3 // question/answer pairs sent with each question (server allows 6 messages)
const HISTORY_CHARS = 1500 // per past answer
const CITATION = /\s*\[\d+(?:\s*,\s*\d+)*\]/g

// The last few completed question/answer pairs of this chat, oldest first, so a follow-up
// like "what about my lawyer?" or "what did you say?" keeps its context. Failed and
// interrupted answers are skipped (refusals are kept: they are part of the conversation),
// and [n] marks are removed: they pointed at that question's passages, not the next one's.
const recentHistory = (messages) => {
  const history = []
  messages.forEach((q, i) => {
    const a = messages[i + 1]
    if (q.role !== 'user' || a?.role !== 'assistant') return
    if (a.pending || a.error || !a.text?.trim()) return
    history.push(
      { role: 'user', text: q.text },
      { role: 'assistant', text: a.text.replace(CITATION, '').trim().slice(0, HISTORY_CHARS) },
    )
  })
  return history.slice(-HISTORY_TURNS * 2)
}

const loadConversations = () => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY))
    if (Array.isArray(saved) && saved.length) {
      return saved.map((c) => ({ ...c, messages: c.messages.map(settle) }))
    }
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
  const [openSource, setOpenSource] = useState(null)
  const endRef = useRef(null)
  const streams = useRef(new Set())

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

  // Stop any answer still streaming when the page goes away.
  useEffect(() => {
    const active = streams.current
    return () => active.forEach((controller) => controller.abort())
  }, [])

  const appendMessage = (id, message, title) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, title: title ?? c.title, messages: [...c.messages, message] } : c,
      ),
    )
  }

  const updateMessage = (convId, msgId, update) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId ? { ...c, messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...update(m) } : m)) } : c,
      ),
    )
  }

  const handleSend = async (text) => {
    const convId = active.id
    const title = active.messages.length === 1 ? text.slice(0, 40) : undefined
    const history = recentHistory(active.messages)
    const msgId = crypto.randomUUID()
    appendMessage(convId, { id: crypto.randomUUID(), role: 'user', text }, title)
    appendMessage(convId, {
      id: msgId,
      role: 'assistant',
      text: '',
      status: 'searching',
      sources: [],
      cited: [],
      followups: [],
      pending: true,
    })
    setLoading(true)

    const update = (fn) => updateMessage(convId, msgId, fn)
    const controller = new AbortController()
    streams.current.add(controller)
    try {
      await streamChat(text, {
        history,
        signal: controller.signal,
        onEvent: (event) => {
          switch (event.type) {
            case 'status':
              update(() => ({ status: event.stage }))
              break
            case 'sources':
              update(() => ({ sources: event.sources }))
              break
            case 'token':
              update((m) => ({ text: m.text + event.text }))
              break
            case 'citations':
              // The answer is complete; follow-up questions arrive a little later.
              update(() => ({ cited: event.cited, pending: false }))
              setLoading(false)
              break
            case 'followups':
              update(() => ({ followups: event.questions }))
              break
            case 'error':
              update(() => ({ error: event.message, pending: false }))
              break
            default:
              break
          }
        },
      })
    } catch (err) {
      if (err.name !== 'AbortError') update(() => ({ error: err.message, pending: false }))
    } finally {
      streams.current.delete(controller)
      update(() => ({ pending: false }))
      setLoading(false)
    }
  }

  const closeSource = useCallback(() => setOpenSource(null), [])

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
                    Here for your <span className={styles.accentWord}>legal</span> queries
                  </h1>
                </div>
              )}

              {messages.map((m, i) => (
                <MessageBubble
                  key={m.id ?? i}
                  message={m}
                  onOpenSource={setOpenSource}
                  onFollowup={handleSend}
                  showFollowups={i === messages.length - 1 && !loading}
                />
              ))}
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

      {openSource && <SourcePanel key={openSource.id} source={openSource} onClose={closeSource} />}
    </div>
  )
}
