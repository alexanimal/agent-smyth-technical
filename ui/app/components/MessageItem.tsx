import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism'
import Avatar from './Avatar'
import useCopyToClipboard from '../hooks/useCopyToClipboard'
import MessageWithViewpoints from './MessageWithViewpoints'

interface MessageItemProps {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: number
  isStreaming?: boolean
  sources?: string[]
  alternativeViewpoint?: string | null
}

// Interface for code component props in ReactMarkdown
interface CodeProps {
  node?: any
  inline?: boolean
  className?: string
  children?: React.ReactNode
}

/**
 * Component to display a single message with appropriate styling
 * Includes support for markdown formatting, code syntax highlighting,
 * and optionally sources and alternative viewpoints for assistant messages
 */
export const MessageItem: React.FC<MessageItemProps> = ({
  role,
  content,
  timestamp,
  isStreaming = false,
  sources = [],
  alternativeViewpoint = null
}) => {
  const [isCopied, copyToClipboard] = useCopyToClipboard();

  // Format timestamp if available
  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      })
    : '';

  // Determine if this is a system message (error, notification, etc.)
  const isSystemMessage = role === 'system';

  // For system messages (errors, notifications)
  if (isSystemMessage) {
    return (
      <div className="flex justify-center my-4 animate-fade-in">
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-3 rounded shadow-md max-w-lg">
          <p className="text-sm">{content}</p>
        </div>
      </div>
    );
  }

  // For user messages
  if (role === 'user') {
    return (
      <div className="flex flex-col items-end mb-4 animate-fade-in">
        <div className="flex items-end">
          <div className="order-2 mx-2 flex flex-col items-end">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-2xl px-4 py-2 max-w-lg shadow-lg hover:shadow-xl transition-shadow duration-300">
              <p className="text-sm whitespace-pre-wrap">{content}</p>
            </div>
            {timestamp && (
              <span className="text-xs text-gray-500 mt-1 mr-2">{formattedTime}</span>
            )}
          </div>
        </div>
      </div>
    );
  }

  // For assistant messages - use MessageWithViewpoints if we have sources or alternativeViewpoint
  if ((sources && sources.length > 0) || alternativeViewpoint) {
    return (
      <MessageWithViewpoints
        content={content}
        sources={sources || []}
        alternativeViewpoint={alternativeViewpoint}
        timestamp={timestamp}
        isStreaming={isStreaming}
      />
    );
  }

  // Standard assistant message with markdown support - no sources or alternatives
  return (
    <div className={`flex mb-4 ${isStreaming ? 'animate-pulse' : 'animate-fade-in'}`}>
      <Avatar />
      <div className="ml-3 flex flex-col w-full max-w-[80%] sm:max-w-[85%] md:max-w-[80%] lg:max-w-[80%]">
        <div className="bg-white dark:bg-gray-800 rounded-2xl px-4 py-3 shadow-md hover:shadow-lg transition-shadow duration-300 markdown-content message-content-large text-justify">
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }: CodeProps) {
                const match = /language-(\w+)/.exec(className || '');
                const codeContent = String(children).replace(/\n$/, '');

                if (!inline && match) {
                  return (
                    <div className="code-block-wrapper">
                      <div className="code-block-toolbar">
                        <button
                          onClick={() => copyToClipboard(codeContent)}
                          className="copy-button"
                          aria-label="Copy code to clipboard"
                        >
                          {isCopied ? 'Copied!' : 'Copy'}
                        </button>
                      </div>
                      <SyntaxHighlighter
                        language={match[1]}
                        style={dracula}
                        PreTag="div"
                        className="rounded my-2 overflow-hidden"
                      >
                        {codeContent}
                      </SyntaxHighlighter>
                    </div>
                  );
                }

                return (
                  <code className={`${className} bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded`} {...props}>
                    {children}
                  </code>
                );
              },
              // Enhance other markdown elements
              p: ({ children }) => <p className="my-2 text-sm text-justify">{children}</p>,
              h1: ({ children }) => <h1 className="text-xl font-bold my-3">{children}</h1>,
              h2: ({ children }) => <h2 className="text-lg font-bold my-2">{children}</h2>,
              ul: ({ children }) => <ul className="list-disc pl-5 my-2">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-5 my-2">{children}</ol>,
              li: ({ children }) => <li className="my-1">{children}</li>,
              blockquote: ({ children }) => (
                <blockquote className="border-l-4 border-gray-300 pl-3 italic my-2">{children}</blockquote>
              ),
              // Add custom styling for tables
              table: ({ children }) => (
                <div className="overflow-x-auto my-4">
                  <table className="min-w-full divide-y divide-gray-300 dark:divide-gray-700">
                    {children}
                  </table>
                </div>
              ),
              thead: ({ children }) => (
                <thead className="bg-gray-100 dark:bg-gray-800">{children}</thead>
              ),
              tbody: ({ children }) => (
                <tbody className="divide-y divide-gray-200 dark:divide-gray-800">{children}</tbody>
              ),
              tr: ({ children }) => <tr>{children}</tr>,
              th: ({ children }) => (
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300 uppercase tracking-wider">
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td className="px-3 py-2 text-sm">{children}</td>
              ),
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
        {timestamp && (
          <span className="text-xs text-gray-500 mt-1 ml-2">{formattedTime}</span>
        )}
      </div>
    </div>
  );
};

export default MessageItem;
