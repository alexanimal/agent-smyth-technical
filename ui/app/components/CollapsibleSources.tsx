import React, { useState, useRef, useEffect } from 'react';
import SourceChip from './SourceChip';

interface CollapsibleSourcesProps {
  sources: string[];
}

/**
 * CollapsibleSources component displays an expandable list of citation sources.
 * It allows hiding/showing sources to keep the UI clean while providing access to references.
 * Now uses a chip-based design for more compact and visually appealing display.
 * Includes mobile-friendly features like limiting initial display and filtering.
 */
export const CollapsibleSources: React.FC<CollapsibleSourcesProps> = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState<number>(0);
  const [expandedSource, setExpandedSource] = useState<string | null>(null);
  const [showAllSources, setShowAllSources] = useState(false);
  const [filter, setFilter] = useState('');

  // Number of sources to show initially (more sources can be revealed with "Show more" button)
  const initialVisibleCount = 10;

  // Toggle the open/closed state
  const toggle = () => setIsOpen(!isOpen);

  // Update content height when sources change or when opened/closed
  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(isOpen ? contentRef.current.scrollHeight : 0);
    }
  }, [isOpen, sources, expandedSource, showAllSources, filter]);

  // Don't render anything if there are no sources
  if (!sources || sources.length === 0) return null;

  // Function to parse and format source URL for display
  const parseSourceUrl = (source: string): { displayUrl: string, url: string } => {
    let displayUrl = '';
    let url = source;

    try {
      const parsedUrl = new URL(source);
      // Handle Twitter/X URLs specially
      if (parsedUrl.hostname === 'x.com' || parsedUrl.hostname.includes('twitter')) {
        const pathParts = parsedUrl.pathname.split('/');
        const username = pathParts[1];
        displayUrl = `@${username}`;
      } else {
        displayUrl = parsedUrl.hostname.replace('www.', '');
      }
    } catch (e) {
      displayUrl = source; // Fall back to the raw source if URL parsing fails
    }

    return { displayUrl, url };
  };

  // Filter sources based on search input
  const filteredSources = filter
    ? sources.filter(source => {
        try {
          const url = new URL(source);
          return url.hostname.toLowerCase().includes(filter.toLowerCase()) ||
                 url.pathname.toLowerCase().includes(filter.toLowerCase());
        } catch {
          return source.toLowerCase().includes(filter.toLowerCase());
        }
      })
    : sources;

  // Decide how many sources to display based on showAllSources state
  const displayedSources = showAllSources
    ? filteredSources
    : filteredSources.slice(0, initialVisibleCount);

  const hasMoreSources = filteredSources.length > initialVisibleCount;

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
        <div className="p-4">
          {/* Search filter for many sources */}
          {sources.length > 15 && (
            <div className="mb-3">
              <input
                type="text"
                placeholder="Filter sources..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="w-full px-3 py-1.5 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-200"
                aria-label="Filter sources"
              />
            </div>
          )}

          {/* Source chips grid layout */}
          <div className="flex flex-wrap gap-1">
            {displayedSources.map((source, index) => {
              const { displayUrl, url } = parseSourceUrl(source);
              return (
                <SourceChip
                  key={index}
                  index={index + 1}
                  url={url}
                  displayText={displayUrl}
                  onClick={() => setExpandedSource(expandedSource === source ? null : source)}
                />
              );
            })}
          </div>

          {/* Show more/less button for many sources */}
          {hasMoreSources && (
            <button
              onClick={() => setShowAllSources(!showAllSources)}
              className="mt-3 text-xs text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 rounded px-2 py-1"
            >
              {showAllSources
                ? "Show fewer sources"
                : `Show ${filteredSources.length - initialVisibleCount} more sources`}
            </button>
          )}

          {/* No results message when filter returns empty */}
          {filter && filteredSources.length === 0 && (
            <div className="my-3 text-sm text-gray-500 dark:text-gray-400">
              No sources match your filter.
            </div>
          )}

          {/* Display expanded source details when a chip is clicked */}
          {expandedSource && (
            <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-md text-sm">
              <div className="font-medium mb-1">Source details:</div>
              <a
                href={expandedSource}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 dark:text-blue-400 hover:underline break-all"
              >
                {expandedSource}
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CollapsibleSources;
