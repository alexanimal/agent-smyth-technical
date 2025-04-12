'use client';

import React, { useState, useEffect } from 'react'
import { MessageList } from './MessageList'
import { InputArea } from './InputArea'
import { SettingsPanel } from './settings/SettingsPanel'
import { useChatStore } from '../store/chatStore'
import { useSettingsStore } from '../store/settingsStore'

/**
 * Main container component for the chat application
 * Manages layout and user interface state
 */
const ChatContainer: React.FC = () => {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const clearMessages = useChatStore(state => state.clearMessages)
  const currentModel = useSettingsStore(state => state.model)
  const currentApiType = useSettingsStore(state => state.apiType)

  // Set the document title to include the current model
  useEffect(() => {
    document.title = `Chat with ${currentModel}`
  }, [currentModel])

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-950" data-testid="chat-container">
      <header className="bg-white dark:bg-gray-900 dark:text-white border-b dark:border-gray-700 p-4 flex justify-between items-center">
        <div className="flex items-center">
          <h1 className="text-xl font-bold">AI Financial Advisor</h1>
          <div className="ml-3 text-xs px-2 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
            {currentApiType === 'stream' ? 'Streaming' : 'Regular'} · {currentModel}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => clearMessages()}
            className="px-3 py-1 text-sm border rounded hover:bg-gray-100 dark:hover:bg-gray-800 dark:border-gray-700 dark:text-gray-200 transition-colors"
            aria-label="Clear chat history"
            data-testid="clear-button"
          >
            Clear Chat
          </button>
          <button
            onClick={() => setSettingsOpen(true)}
            className="px-3 py-1 text-sm bg-gray-800 text-white rounded hover:bg-gray-700 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors"
            aria-label="Open settings"
            data-testid="settings-button"
          >
            Settings
          </button>
        </div>
      </header>

      <MessageList />
      <InputArea />

      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}

export default ChatContainer
