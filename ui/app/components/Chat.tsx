import React from 'react';
import { MessageList } from './MessageList';
import { InputArea } from './InputArea';
import LoadingIndicator from './LoadingIndicator';
import { useChatStore } from '../store/chatStore';

/**
 * Main Chat component that combines all chat-related components
 * Includes header, message list, and input area
 */
const Chat: React.FC = () => {
  const { isLoading, clearMessages } = useChatStore();

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      {/* Header */}
      <header className="border-b dark:border-gray-700 p-4 bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-semibold bg-gradient-to-r from-blue-500 to-indigo-600 inline-block text-transparent bg-clip-text">
            Chat Application
          </h1>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {isLoading ? 'Processing...' : 'Ready'}
            </span>
            {isLoading && <LoadingIndicator size="small" />}

            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all messages? This cannot be undone.')) {
                  clearMessages();
                }
              }}
              className="text-sm px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded-md transition-colors"
              aria-label="Clear chat history"
            >
              Clear Chat
            </button>
          </div>
        </div>
      </header>

      {/* Message List - Flex grow to take available space */}
      <div className="flex-1 overflow-hidden relative">
        <MessageList />
      </div>

      {/* Input Area - Fixed at bottom */}
      <InputArea />
    </div>
  );
};

export default Chat;
