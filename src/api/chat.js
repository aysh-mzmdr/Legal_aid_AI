// Client for the RAG server in "RAG model/" (`uv run rag-serve`).
// Set VITE_API_URL in a .env file to point somewhere other than the default.
export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export class ChatError extends Error {
  constructor(message, { retryable = false } = {}) {
    super(message)
    this.retryable = retryable
  }
}

// Streams an answer. The server sends one JSON event per line (see answer.py):
// status, sources, token, citations, followups, done, error. onEvent gets each one.
export async function streamChat(question, { onEvent, signal }) {
  let response
  try {
    response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new ChatError("Can't reach the assistant. Is the server running? (uv run rag-serve)", { retryable: true })
  }

  if (!response.ok) {
    let detail = ''
    try {
      detail = (await response.json()).detail
    } catch {
      // not JSON
    }
    if (response.status === 503) {
      throw new ChatError('The assistant is still starting up. Try again in a moment.', { retryable: true })
    }
    throw new ChatError(typeof detail === 'string' && detail ? detail : `The server returned an error (${response.status}).`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
    const lines = buffer.split('\n')
    buffer = lines.pop()
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line))
    }
    if (done) break
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer))
}

const provisionCache = new Map()

// Full text of a cited provision, for the source panel. Cached for the session.
export async function fetchProvision(id) {
  if (!provisionCache.has(id)) {
    const request = fetch(`${API_URL}/api/provisions/${encodeURIComponent(id)}`).then(async (r) => {
      if (!r.ok) throw new ChatError(r.status === 503 ? 'The assistant is still starting up.' : 'Could not load this provision.')
      return r.json()
    })
    provisionCache.set(id, request)
    request.catch(() => provisionCache.delete(id))
  }
  return provisionCache.get(id)
}
