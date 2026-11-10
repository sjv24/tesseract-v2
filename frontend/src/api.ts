import type { ChatRequest, ChatResponse, StreamEvent } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }
  return response.json()
}

// Turn one server-sent event block ("event: x" + "data: {...}") into a StreamEvent.
export function parseEvent(block: string): StreamEvent | null {
  const name = /^event: (.+)$/m.exec(block)?.[1]
  const data = /^data: (.+)$/m.exec(block)?.[1]
  if (!name || !data) return null
  return { type: name, ...JSON.parse(data) } as StreamEvent
}

// Ask a question and receive the answer piece by piece. onEvent is called for every event as it arrives.
export async function streamChat(
  request: ChatRequest,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok || !response.body) {
    throw new Error(await errorMessage(response))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // events are separated by a blank line; keep any unfinished event in the buffer
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      const event = parseEvent(block)
      if (event) onEvent(event)
    }
  }
}

// The backend explains problems in {"detail": "..."}; fall back to the status code.
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
  } catch {
    // not JSON
  }
  return `Request failed (${response.status})`
}
