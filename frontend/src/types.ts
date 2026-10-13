export type ChatRequest = {
  question: string
}

export type ChatResponse = {
  answer: string
  category: 'script' | 'textbook' | 'comments'
}