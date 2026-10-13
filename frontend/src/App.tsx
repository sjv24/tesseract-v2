import { useState } from 'react'
import { sendChat } from './api'
import type { ChatResponse } from './types'

type Message = ChatResponse & { question: string }

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSend() {
    const text = question.trim()
    if (!text || loading) return

    setLoading(true)
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
    <div>
      <h1>Tesseract</h1>

      {messages.map((m, i) => (
        <div key={i}>
          <p><strong>You:</strong> {m.question}</p>
          <p><strong>Tesseract ({m.category}):</strong> {m.answer}</p>
        </div>
      ))}

      {loading && <p>Thinking... this can take up to 30 seconds</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        placeholder="Ask about Interstellar..."
        disabled={loading}
      />
      <button onClick={handleSend} disabled={loading}>Send</button>
    </div>
  )
}

export default App