import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';
import CollapsibleSources from './CollapsibleSources';
import ViewpointToggle from './ViewpointToggle';
import Avatar from './Avatar';

interface MessageWithViewpointsProps {
  content: string;
  alternativeViewpoint: string | null;
  sources: string[];
  timestamp?: number;
  isStreaming?: boolean;
}

// Interface for code component props in ReactMarkdown
interface CodeProps {
  node?: any;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

/**
 * Enhanced message component that supports:
 * 1. Toggling between main and alternative viewpoints
 * 2. Displaying collapsible sources/citations
 * 3. Markdown rendering with syntax highlighting
 */
export const MessageWithViewpoints: React.FC<MessageWithViewpointsProps> = ({
  content,
  alternativeViewpoint,
  sources,
  timestamp,
  isStreaming = false
}) => {
  const [showAlternative, setShowAlternative] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  // Check for valid sources and alternative viewpoint data
  const hasValidSources = Array.isArray(sources) && sources.length > 0;
  const hasAlternative = typeof alternativeViewpoint === 'string' && alternativeViewpoint.trim().length > 0;

  // Current content to display based on toggle state
  const currentContent = showAlternative && hasAlternative ? alternativeViewpoint : content;

  // Debug logs to track props changes
  useEffect(() => {
    console.log('MessageWithViewpoints props updated:', {
      contentLength: content.length,
      isStreaming,
      hasSources: hasValidSources,
      sourcesCount: sources?.length || 0,
      hasAlternativeViewpoint: hasAlternative,
      alternativeViewpointLength: alternativeViewpoint?.length || 0,
      showingAlternative: showAlternative
    });
  }, [content, isStreaming, sources, alternativeViewpoint, showAlternative, hasValidSources, hasAlternative]);

  // Format timestamp if available
  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      })
    : '';

  // Copy code to clipboard
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  // Define streaming animation class
  const streamingAnimationClass = isStreaming
    ? 'border-l-4 border-blue-500 pl-3 animate-pulse-subtle'
    : 'animate-fade-in';

  return (
    <div className="message-with-viewpoints transition-all duration-300">
      {/* Only show viewpoint toggle when we have a valid alternative viewpoint */}
      {hasAlternative && (
        <ViewpointToggle
          showAlternative={showAlternative}
          onToggle={setShowAlternative}
          hasAlternative={hasAlternative}
        />
      )}

      <div className={`flex mb-4 ${streamingAnimationClass}`}>
        <Avatar isAnimated={isStreaming} />
        <div className="ml-3 flex flex-col max-w-[80%] sm:max-w-[85%] md:max-w-[80%] lg:max-w-[80%]">
          <div className={`w-fit bg-white dark:bg-gray-800 rounded-2xl px-4 py-3 shadow-md hover:shadow-lg transition-shadow duration-300 markdown-content message-content-large text-left`}>
            {/* Streaming indicator */}
            {isStreaming && (
              <div className="flex items-center mb-2 text-blue-500 text-xs font-medium">
                <div className="mr-2 flex space-x-1">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bouncing-dot" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bouncing-dot" style={{ animationDelay: '300ms' }}></div>
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bouncing-dot" style={{ animationDelay: '600ms' }}></div>
                </div>
                <span>Generating response</span>
              </div>
            )}

            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }: CodeProps) {
                  const match = /language-(\w+)/.exec(className || '');
                  const codeContent = String(children).replace(/\n$/, '');

                  if (!inline && match) {
                    return (
                      <div className="code-block-wrapper relative rounded overflow-hidden my-3 bg-gray-900">
                        <div className="code-block-header flex items-center justify-between px-4 py-2 bg-gray-800 text-gray-400 text-xs">
                          <span>{match[1]}</span>
                          <button
                            onClick={() => copyToClipboard(codeContent)}
                            className="copy-button px-2 py-1 rounded hover:bg-gray-700 transition-colors duration-200"
                            aria-label="Copy code to clipboard"
                          >
                            {isCopied ? 'Copied!' : 'Copy'}
                          </button>
                        </div>
                        <SyntaxHighlighter
                          language={match[1]}
                          style={dracula}
                          customStyle={{ margin: 0, borderRadius: 0 }}
                          showLineNumbers={codeContent.split('\n').length > 3}
                          wrapLongLines={false}
                        >
                          {codeContent}
                        </SyntaxHighlighter>
                      </div>
                    );
                  }

                  return (
                    <code className={`${className} bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded font-mono text-sm`} {...props}>
                      {children}
                    </code>
                  );
                },
                // Enhance other markdown elements
                p: ({ children }) => <p className="my-2 text-sm text-left leading-relaxed">{children}</p>,
                h1: ({ children }) => <h1 className="text-xl font-bold my-3 border-b pb-1 border-gray-200 dark:border-gray-700 text-left">{children}</h1>,
                h2: ({ children }) => <h2 className="text-lg font-bold my-2 text-gray-800 dark:text-gray-200 text-left">{children}</h2>,
                h3: ({ children }) => <h3 className="text-md font-semibold my-2 text-gray-800 dark:text-gray-200 text-left">{children}</h3>,
                ul: ({ children }) => <ul className="list-disc pl-5 my-2 space-y-1 text-left">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 my-2 space-y-1 text-left">{children}</ol>,
                li: ({ children }) => <li className="my-1 text-sm text-left">{children}</li>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-3 italic my-3 text-gray-700 dark:text-gray-300 text-left">{children}</blockquote>
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
                  <td className="px-3 py-2 text-sm text-left">{children}</td>
                ),
              }}
            >
              {currentContent}
            </ReactMarkdown>
          </div>

          {timestamp && !isStreaming && (
            <span className="text-xs text-gray-500 mt-1 ml-2">{formattedTime}</span>
          )}

          {/* Only show sources if we have valid sources */}
          {hasValidSources && (
            <CollapsibleSources sources={sources} />
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageWithViewpoints;
