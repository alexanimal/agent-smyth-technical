import React, { useState, useRef, useEffect } from 'react'
import { useChat } from '../hooks/useChat'
import { useSettingsStore } from '../store/settingsStore'

/**
 * Input area component for text entry and message submission
 * Features expandable text area and settings button
 */
export const InputArea: React.FC = () => {
  const [inputValue, setInputValue] = useState('')
  const [isExpanded, setIsExpanded] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const settingsRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { sendMessage, isLoading, cancelRequest } = useChat()
  const settings = useSettingsStore()

  // Handle clicking outside settings menu to close it
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setShowSettings(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  // Auto-resize the textarea as content changes
  useEffect(() => {
    if (textareaRef.current && isExpanded) {
      textareaRef.current.style.height = 'inherit'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [inputValue, isExpanded])

  // Handle sending message
  const handleSendMessage = () => {
    if (inputValue.trim() && !isLoading) {
      sendMessage(inputValue)
      setInputValue('')
      setIsExpanded(false)

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'inherit'
      }
    }
  }

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Submit on Enter (but not with Shift+Enter which adds a new line)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }

    // Expand on Shift+Enter
    if (e.key === 'Enter' && e.shiftKey && !isExpanded) {
      setIsExpanded(true)
    }

    // Cancel request with Escape
    if (e.key === 'Escape' && isLoading) {
      cancelRequest()
    }
  }

  return (
    <div className="border-t dark:border-gray-700 p-4 bg-white dark:bg-gray-900">
      <div className="relative max-w-4xl mx-auto">
        {isExpanded ? (
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Shift+Enter for new line, Esc to cancel)"
            className={`w-full p-3 pr-16 rounded-lg border ${isLoading ? 'bg-gray-100 dark:bg-gray-800' : 'bg-white dark:bg-gray-950'}
                       dark:text-white border-gray-300 dark:border-gray-700 shadow-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                       transition-all duration-200 ease-in-out resize-none`}
            rows={3}
            disabled={isLoading}
          />
        ) : (
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => inputValue.length > 50 && setIsExpanded(true)}
            placeholder="Type your message..."
            className={`w-full p-3 pr-16 rounded-full border ${isLoading ? 'bg-gray-100 dark:bg-gray-800' : 'bg-white dark:bg-gray-950'}
                      dark:text-white border-gray-300 dark:border-gray-700 shadow-sm
                      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                      transition-all duration-200 ease-in-out`}
            disabled={isLoading}
          />
        )}

        <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
          {/* Settings button */}
          <div ref={settingsRef} className="relative">
            <button
              type="button"
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800"
              disabled={isLoading}
              aria-label="Settings"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
              </svg>
            </button>

            {showSettings && (
              <div className="absolute bottom-12 right-0 w-64 bg-white dark:bg-gray-800 shadow-lg rounded-lg p-4 border border-gray-200 dark:border-gray-700 z-10">
                <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">Chat Settings</h3>

                {/* API Type Toggle */}
                <div className="mb-3">
                  <label className="text-xs font-medium text-gray-700 dark:text-gray-300">API Type</label>
                  <div className="flex mt-1 rounded-md overflow-hidden border border-gray-300 dark:border-gray-700">
                    <button
                      className={`flex-1 py-1 text-xs ${settings.apiType === 'regular' ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-850 text-gray-700 dark:text-gray-300'}`}
                      onClick={() => settings.setApiType('regular')}
                    >
                      Regular
                    </button>
                    <button
                      className={`flex-1 py-1 text-xs ${settings.apiType === 'stream' ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-850 text-gray-700 dark:text-gray-300'}`}
                      onClick={() => settings.setApiType('stream')}
                    >
                      Streaming
                    </button>
                  </div>
                </div>

                {/* Model Selector */}
                <div className="mb-3">
                  <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Model</label>
                  <select
                    value={settings.model}
                    onChange={(e) => settings.setModel(e.target.value)}
                    className="mt-1 block w-full py-1 px-2 border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 rounded-md shadow-sm text-xs focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:text-white"
                  >
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                  </select>
                </div>

                {/* Number of Records */}
                <div className="mb-3">
                  <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Records: {settings.numRecords}</label>
                  <input
                    type="range"
                    min="5"
                    max="250"
                    step="5"
                    value={settings.numRecords}
                    onChange={(e) => settings.setNumRecords(Number(e.target.value))}
                    className="w-full h-2 mt-1 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>5</span>
                    <span>250</span>
                  </div>
                </div>

                {/* Alternative Opinions */}
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Alternative Opinions</label>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.generateAlternativeOpinions}
                      onChange={(e) => settings.setGenerateAlternativeOpinions(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-blue-500"></div>
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Send button */}
          <button
            type="button"
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className={`p-2 rounded-full ${
              !inputValue.trim() || isLoading
                ? 'bg-gray-200 text-gray-400 dark:bg-gray-800 dark:text-gray-600 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-md hover:shadow-lg hover:from-blue-600 hover:to-blue-700'
            } transition-all duration-200`}
            aria-label="Send message"
          >
            {isLoading ? (
              <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
