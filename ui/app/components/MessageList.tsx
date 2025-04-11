import React, { useRef, useEffect } from 'react'
import { MessageItem } from './MessageItem'
import { useChatStore } from '../store/chatStore'

/**
 * Component that renders a list of message items
 * Automatically scrolls to the bottom when new messages are added
 */
export const MessageList: React.FC = () => {
  const { messages, streamingMessage, isLoading } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when messages change or streaming content updates
  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingMessage])

  // Function to handle scrolling to the bottom of the chat
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      // Try using scrollIntoView first
      try {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
      } catch (error) {
        // Fallback to manual scrolling if scrollIntoView fails
        if (listRef.current) {
          listRef.current.scrollTop = listRef.current.scrollHeight
        }
      }
    }
  }

  return (
    <div
      ref={listRef}
      className="flex flex-col p-4 space-y-4 overflow-y-auto h-full pb-6 scrollbar-thin"
    >
      {/* Welcome message if no messages yet */}
      {messages.length === 0 && (
        <div className="text-center py-8 animate-fade-in">
          <h2 className="text-lg font-medium mb-2">Welcome to the Chat!</h2>
          <p className="text-gray-500 dark:text-gray-400">
            Send a message to start a conversation.
          </p>
        </div>
      )}

      {/* Render all messages */}
      <div className="flex flex-col space-y-6">
        {messages.map((message) => (
          <MessageItem
            key={message.id}
            role={message.role}
            content={message.content}
            timestamp={message.timestamp}
            sources={message.sources || []}
            alternativeViewpoint={message.alternativeViewpoint || null}
          />
        ))}

        {/* Show streaming message if any */}
        {streamingMessage && (
          <MessageItem
            role="assistant"
            content={streamingMessage}
            isStreaming={true}
            sources={[]} // No sources during streaming
            alternativeViewpoint={null} // No alternative during streaming
          />
        )}
      </div>

      {/* Empty div for scrolling to bottom */}
      <div ref={messagesEndRef} />

      {/* Small space at bottom to ensure messages don't get hidden behind input */}
      <div className="h-4" />
    </div>
  )
}

export default MessageList
