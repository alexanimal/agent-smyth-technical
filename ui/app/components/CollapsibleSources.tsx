import React, { useState, useRef, useEffect } from 'react';

interface CollapsibleSourcesProps {
  sources: string[];
}

/**
 * CollapsibleSources component displays an expandable list of citation sources.
 * It allows hiding/showing sources to keep the UI clean while providing access to references.
 */
export const CollapsibleSources: React.FC<CollapsibleSourcesProps> = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState<number>(0);

  // Toggle the open/closed state
  const toggle = () => setIsOpen(!isOpen);

  // Update content height when sources change or when opened/closed
  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(isOpen ? contentRef.current.scrollHeight : 0);
    }
  }, [isOpen, sources]);

  // Don't render anything if there are no sources
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-container mt-4 border-t border-gray-200 dark:border-gray-700">
      <button
        onClick={toggle}
        className="sources-toggle w-full px-4 py-2 text-left text-sm flex justify-between items-center text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors duration-200"
        aria-expanded={isOpen}
        aria-controls="sources-content"
      >
        <span>Sources ({sources.length})</span>
        <span className="transform transition-transform duration-200" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
          ▼
        </span>
      </button>

      <div
        id="sources-content"
        ref={contentRef}
        className="sources-content overflow-hidden transition-all duration-300 ease-in-out"
        style={{ height: `${contentHeight}px` }}
        aria-hidden={!isOpen}
      >
        <ol className="sources-list px-6 py-3 text-sm text-gray-600 dark:text-gray-300 space-y-2">
          {sources.map((source, index) => {
            // Parse URL to display in a more readable format
            let displayUrl = '';
            try {
              const url = new URL(source);
              // Handle Twitter/X URLs specially
              if (url.hostname === 'x.com') {
                const pathParts = url.pathname.split('/');
                const username = pathParts[1];
                displayUrl = `@${username} on Twitter (X)`;
              } else {
                displayUrl = url.hostname.replace('www.', '');
              }
            } catch (e) {
              displayUrl = source; // Fall back to the raw source if URL parsing fails
            }

            return (
              <li key={index} className="source-item">
                <a
                  href={source}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-blue-500 transition-colors duration-200"
                >
                  [{index + 1}] {displayUrl}
                </a>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
};

export default CollapsibleSources;
