import React from 'react';

interface ViewpointToggleProps {
  showAlternative: boolean;
  onToggle: (showAlternative: boolean) => void;
  hasAlternative: boolean;
}

/**
 * ViewpointToggle component provides a UI to switch between
 * the main analysis view and an alternative viewpoint.
 * Only displays when an alternative viewpoint is available.
 */
export const ViewpointToggle: React.FC<ViewpointToggleProps> = ({
  showAlternative,
  onToggle,
  hasAlternative
}) => {
  if (!hasAlternative) return null;

  return (
    <div className="viewpoint-toggle flex border-b border-gray-200 dark:border-gray-700 mb-4">
      <button
        className={`py-2 px-4 text-sm font-medium ${
          !showAlternative
            ? 'text-blue-600 border-b-2 border-blue-600 dark:text-blue-500 dark:border-blue-500'
            : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
        }`}
        onClick={() => onToggle(false)}
        aria-pressed={!showAlternative}
      >
        Main Analysis
      </button>
      <button
        className={`py-2 px-4 text-sm font-medium ${
          showAlternative
            ? 'text-blue-600 border-b-2 border-blue-600 dark:text-blue-500 dark:border-blue-500'
            : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
        }`}
        onClick={() => onToggle(true)}
        aria-pressed={showAlternative}
      >
        Alternative Viewpoint
      </button>
    </div>
  );
};

export default ViewpointToggle;
