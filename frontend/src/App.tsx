import { useEffect, useRef, useState } from 'react'
import { sendChat } from './api'
import type { ChatResponse } from './types'
import './App.css'

type Message = ChatResponse & { question: string }

const SOURCE_LABELS: Record<ChatResponse['category'], string> = {
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
    try {
      const response = await sendChat({ question: text })
      setMessages((prev) => [...prev, { question: text, ...response }])
      setQuestion('')
    } catch (err) {
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
              <span className="badge">{SOURCE_LABELS[m.category]}</span>
              <p>{m.answer}</p>
            </div>
          </div>
        ))}

        {loading && (
          <div className="status" aria-live="polite">
            Thinking... {elapsed}s (answers can take up to a minute)
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
