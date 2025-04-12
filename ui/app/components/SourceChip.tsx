import React from 'react';

interface SourceChipProps {
  index: number;
  url: string;
  displayText: string;
  onClick?: () => void;
}

/**
 * SourceChip component displays a single source reference as a compact chip
 * Used within the CollapsibleSources component to provide a more space-efficient display
 */
const SourceChip: React.FC<SourceChipProps> = ({ index, url, displayText, onClick }) => {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(e) => {
        if (onClick) {
          e.preventDefault();
          onClick();
        }
      }}
      className="inline-flex items-center px-3 py-1 m-1 rounded-full bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-sm text-gray-800 dark:text-gray-200 transition-colors"
    >
      <span className="font-medium mr-1">[{index}]</span>
      <span className="truncate max-w-[150px]">{displayText}</span>
    </a>
  );
};

export default SourceChip;
