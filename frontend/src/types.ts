export type Category = 'script' | 'textbook' | 'comments' | 'multi'

export type ChatRequest = {
  question: string
}

// One retrieved passage the answer was based on.
export type Citation = {
  source: 'script' | 'textbook' | 'comments'
  collection: string
  chunk_id: string
  section: string | null
  page: number | null
  score: number
  snippet: string
}

// What /chat returns in one piece.
export type ChatResponse = {
  answer: string
  category: Category
  citations: Citation[]
}

// What /chat/stream sends, one event at a time.
export type StreamEvent =
  | { type: 'meta'; category: Category; citations: Citation[] }
  | { type: 'token'; text: string }
  | { type: 'done'; seconds: number }
  | { type: 'error'; detail: string }
