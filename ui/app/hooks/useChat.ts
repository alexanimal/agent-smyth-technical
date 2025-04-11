import { useCallback, useRef, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { useChatStore } from '../store/chatStore'
import { useSettingsStore } from '../store/settingsStore'
import { sendMessage, processStreamResponse } from '../api/chatAPI'
import { ChatMessage } from '../store/types'

/**
 * Custom hook for managing chat functionality
 * Handles sending messages, streaming responses, and cancellation
 */
export const useChat = () => {
  const {
    messages,
    addMessage,
    setStreamingMessage,
    finalizeStreamingMessage,
    setLoading,
    isLoading,
    streamingMessage
  } = useChatStore()

  const settings = useSettingsStore()
  const abortControllerRef = useRef<AbortController | null>(null)

  // For debugging purposes, log streaming message changes
  useEffect(() => {
    console.log('Streaming message updated:', streamingMessage)
  }, [streamingMessage])

  /**
   * Sends a user message to the chat API
   * Handles both streaming and regular responses
   */
  const sendUserMessage = useCallback(async (content: string) => {
    if (!content.trim()) return

    // Create user message
    const userMessage: ChatMessage = {
      role: 'user',
      content,
      id: uuidv4(),
      timestamp: Date.now()
    }
    addMessage(userMessage)
    setLoading(true)

    try {
      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController()

      console.log('Sending message to server:', content)
      const response = await sendMessage({
        message: content,
        settings,
        abortController: abortControllerRef.current
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      if (settings.apiType === 'stream') {
        // Handle streaming response
        console.log('Processing streaming response')
        let fullContent = ''

        try {
          for await (const chunk of processStreamResponse(response)) {
            console.log('Received chunk:', chunk)
            if (chunk) {
              fullContent += chunk
              console.log('Updated streaming content:', fullContent)
              setStreamingMessage(fullContent)
            }
          }
          console.log('Stream processing completed, finalizing message with content:', fullContent)
          if (fullContent.trim()) {
            setStreamingMessage(fullContent)
            finalizeStreamingMessage()
          } else {
            console.warn('No content received from stream')
            addMessage({
              role: 'system',
              content: 'No content received from the streaming API.',
              id: uuidv4(),
              timestamp: Date.now()
            })
          }
        } catch (streamError) {
          console.error('Error processing stream:', streamError)
          setStreamingMessage(null)
          addMessage({
            role: 'system',
            content: `Error processing stream: ${streamError}`,
            id: uuidv4(),
            timestamp: Date.now()
          })
        }
      } else {
        // Handle regular response
        console.log('Processing regular response')
        try {
          const data = await response.json()
          console.log('Received response data:', data)
          addMessage({
            role: 'assistant',
            content: data.response || data.content || 'No content received',
            id: uuidv4(),
            timestamp: Date.now(),
            sources: data.sources || [],
            alternativeViewpoint: data.alternative_viewpoints || null
          })
        } catch (jsonError) {
          console.error('Error parsing JSON response:', jsonError)
          addMessage({
            role: 'system',
            content: `Error parsing response: ${jsonError}`,
            id: uuidv4(),
            timestamp: Date.now()
          })
        }
      }
    } catch (error: any) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        console.error('Error sending message:', error)
        addMessage({
          role: 'system',
          content: `Error: ${error.message || 'Failed to send message. Please try again.'}`,
          id: uuidv4(),
          timestamp: Date.now()
        })
      }
    } finally {
      setLoading(false)
      abortControllerRef.current = null
    }
  }, [addMessage, setStreamingMessage, finalizeStreamingMessage, setLoading, settings])

  /**
   * Cancels the current request if one is in progress
   */
  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setLoading(false)
      setStreamingMessage(null)
    }
  }, [setLoading, setStreamingMessage])

  return {
    messages,
    sendMessage: sendUserMessage,
    cancelRequest,
    isLoading,
    streamingMessage,
  }
}
