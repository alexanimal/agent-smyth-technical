import { SendMessageParams } from '../store/types'

// Define the base URL for our chat API
const API_BASE_URL = 'http://localhost:8003'
const REGULAR_ENDPOINT = `${API_BASE_URL}/chat`
const STREAMING_ENDPOINT = `${API_BASE_URL}/chat/stream`

/**
 * Sends a chat message to the appropriate API endpoint based on settings
 */
export const sendMessage = async ({
  message,
  settings,
  abortController
}: SendMessageParams): Promise<Response> => {
  const endpoint = settings.apiType === 'stream'
    ? STREAMING_ENDPOINT
    : REGULAR_ENDPOINT

  // Prepare the request body including num_records if available
  const requestBody = {
    message,
    model: settings.model,
    generate_alternative_viewpoint: settings.generateAlternativeOpinions,
    num_results: settings.numRecords || 50 // Default to 50 if not set
  }

  return fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': '1234567890'
    },
    body: JSON.stringify(requestBody),
    signal: abortController?.signal
  })
}

/**
 * Processes a streaming response from the server
 * Yields chunks of text as they arrive
 */
export async function* processStreamResponse(
  response: Response
): AsyncGenerator<string> {
  const reader = response.body?.getReader()
  if (!reader) throw new Error('Response body is null')

  const decoder = new TextDecoder()
  console.log('Starting to process stream response')

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        console.log('Stream completed')
        break
      }

      const chunk = decoder.decode(value, { stream: true })
      console.log('Raw chunk received:', chunk)

      // Split by event lines to handle different event types
      const eventChunks = chunk.split('event: ').filter(Boolean);

      for (const eventChunk of eventChunks) {
        const eventLines = eventChunk.split('\n');
        const eventType = eventLines[0]?.trim();

        // Find the data line
        const dataLine = eventLines.find(line => line.startsWith('data:'));
        if (!dataLine) continue;

        const data = dataLine.substring(dataLine.indexOf('data:') + 5).trim();

        if (data === '[DONE]') {
          console.log('Received [DONE] marker');
          continue;
        }

        if (data) {
          try {
            const parsed = JSON.parse(data);
            console.log('Parsed JSON for event:', eventType, parsed);

            // Handle based on event type
            if (eventType === 'chunk' && parsed.text) {
              console.log('Yielding chunk text:', parsed.text);
              yield parsed.text;
            } else if (eventType === 'complete' && parsed.response) {
              // If we're getting complete event, we already yielded the chunks
              // Just log it for debugging purposes
              console.log('Received complete response:', parsed.response);
            } else {
              // Fallback to trying the original fields or other fields that might contain content
              const content =
                parsed.content ||
                parsed.text ||
                parsed.response ||
                parsed.choices?.[0]?.delta?.content ||
                parsed.choices?.[0]?.text ||
                '';

              if (content) {
                console.log('Yielding content from fallback:', content);
                yield content;
              }
            }
          } catch (e) {
            console.error('Failed to parse SSE chunk:', e, 'Line was:', data);
            // Try to extract content even if JSON parsing fails
            if (data !== '[DONE]') {
              yield data;
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
    console.log('Stream reader released')
  }
}
