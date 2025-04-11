import React from 'react';

interface LoadingIndicatorProps {
  size?: 'small' | 'medium' | 'large';
}

/**
 * Animated loading indicator with bouncing dots
 * Can be used throughout the app to provide consistent loading feedback
 */
const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ size = 'medium' }) => {
  // Determine dot sizes based on component size
  const dotSizes = {
    small: 'h-1 w-1',
    medium: 'h-2 w-2',
    large: 'h-3 w-3',
  };

  return (
    <div className="flex space-x-2 items-center justify-center my-2" aria-label="Loading" role="status">
      <div
        className={`bg-blue-500 rounded-full animate-bounce ${dotSizes[size]}`}
        style={{ animationDelay: "0s" }}
      />
      <div
        className={`bg-blue-500 rounded-full animate-bounce ${dotSizes[size]}`}
        style={{ animationDelay: "0.2s" }}
      />
      <div
        className={`bg-blue-500 rounded-full animate-bounce ${dotSizes[size]}`}
        style={{ animationDelay: "0.4s" }}
      />
    </div>
  );
};

export default LoadingIndicator;
