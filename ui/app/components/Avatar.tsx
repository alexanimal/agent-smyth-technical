import React from 'react';

interface AvatarProps {
  isAnimated?: boolean;
}

/**
 * Avatar component for the assistant
 * Displays a simple icon or image that represents the AI
 * Can be animated to show the assistant is actively generating
 */
const Avatar: React.FC<AvatarProps> = ({ isAnimated = false }) => {
  const animationClass = isAnimated ? 'shadow-md shadow-blue-300 animate-avatar-pulse' : '';

  return (
    <div className={`w-10 h-10 rounded-full bg-gradient-to-r from-blue-400 to-indigo-500 flex items-center justify-center flex-shrink-0 transition-all duration-300 ${animationClass}`}>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={isAnimated ? 'animate-pulse' : ''}
      >
        <path d="M12 2a8 8 0 0 1 8 8v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V10a8 8 0 0 1 8-8z"></path>
        <path d="M9.5 2A12.5 12.5 0 0 0 12 21.9c1.7.1 3.4 0 5-.3"></path>
        <path d="M8 14v.5"></path>
        <path d="M16 14v.5"></path>
        <path d="M11.5 17h1"></path>
      </svg>
    </div>
  );
};

export default Avatar;
