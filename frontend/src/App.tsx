import { useEffect, useRef, useState } from 'react'
import { streamChat } from './api'
import type { Category, Citation } from './types'
import './App.css'

type Message = { question: string; category?: Category; answer: string; citations: Citation[] }

const SOURCE_LABELS: Record<Category, string> = {
  script: 'Screenplay',
  textbook: 'Science textbook',
  comments: 'Viewer comments',
  multi: 'Multiple sources',
}

const EXAMPLES = [
  'Who is Dr. Mann?',
  'How does a wormhole work?',
  'What do fans think about the ending?',
  'Is the tesseract scene scientifically accurate?',
]

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, error])

  useEffect(() => {
    if (!loading) return
    const id = setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [loading])

  async function ask(raw: string) {
    const text = raw.trim()
    if (!text || loading) return

    setLoading(true)
    setElapsed(0)
    setError(null)
    setQuestion('')
    // add the question with an empty answer now; the answer fills in as pieces arrive
    setMessages((prev) => [...prev, { question: text, answer: '', citations: [] }])
    const updateLast = (change: (m: Message) => Message) =>
      setMessages((prev) => prev.map((m, i) => (i === prev.length - 1 ? change(m) : m)))
    try {
      await streamChat({ question: text }, (event) => {
        if (event.type === 'meta') {
          updateLast((m) => ({ ...m, category: event.category, citations: event.citations }))
        } else if (event.type === 'token') {
          updateLast((m) => ({ ...m, answer: m.answer + event.text }))
        } else if (event.type === 'error') {
          setError(event.detail)
        }
      })
    } catch (err) {
      setMessages((prev) => prev.slice(0, -1)) // drop the unanswered question
      setQuestion(text)
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Tesseract</h1>
        <p>Ask anything about Interstellar</p>
      </header>

      <main className="chat">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <p>Try one of these:</p>
            {EXAMPLES.map((example) => (
              <button key={example} className="example" onClick={() => ask(example)}>
                {example}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className="exchange">
            <div className="bubble user">{m.question}</div>
            <div className="bubble bot">
              {m.category && <span className="badge">{SOURCE_LABELS[m.category]}</span>}
              <p>{m.answer}</p>
              {m.citations.length > 0 && (
                <details className="sources">
                  <summary>Sources ({m.citations.length})</summary>
                  <ol>
                    {m.citations.map((c) => (
                      <li key={c.chunk_id}>
                        <strong>{c.section ?? SOURCE_LABELS[c.source]}</strong>
                        {c.page !== null && <span> · page {c.page}</span>}
                        <div className="snippet">{c.snippet}</div>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="status" aria-live="polite">
            {messages[messages.length - 1]?.answer ? 'Writing' : 'Thinking'}... {elapsed}s
          </div>
        )}
        {error && (
          <div className="status error" role="alert">
            Could not get an answer: {error}
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          ask(question)
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about Interstellar..."
          disabled={loading}
          maxLength={1000}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
