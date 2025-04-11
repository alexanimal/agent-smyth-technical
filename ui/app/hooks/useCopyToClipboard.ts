import { useState } from 'react';

/**
 * Custom hook for copying text to clipboard with success/error feedback
 * @returns Tuple containing isCopied state and copy function
 */
export const useCopyToClipboard = (): [boolean, (text: string) => Promise<boolean>] => {
  const [isCopied, setIsCopied] = useState(false);

  const copy = async (text: string): Promise<boolean> => {
    // Reset the copied state
    setIsCopied(false);

    // If there's nothing to copy, return false
    if (!text) return false;

    // Try to use the Clipboard API
    try {
      if (navigator.clipboard && window.isSecureContext) {
        // The Clipboard API is only available in secure contexts (HTTPS)
        await navigator.clipboard.writeText(text);
        setIsCopied(true);
        // Reset after 2 seconds
        setTimeout(() => setIsCopied(false), 2000);
        return true;
      } else {
        // Fallback for non-secure contexts
        copyUsingTextArea(text);
        setIsCopied(true);
        // Reset after 2 seconds
        setTimeout(() => setIsCopied(false), 2000);
        return true;
      }
    } catch (error) {
      console.error('Failed to copy text: ', error);
      return false;
    }
  };

  // Fallback method for non-secure contexts
  const copyUsingTextArea = (text: string): boolean => {
    // Create a temporary textarea element to hold the text
    const textArea = document.createElement('textarea');
    textArea.value = text;

    // Make the textarea out of viewport
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);

    // Select and copy
    textArea.focus();
    textArea.select();

    let success = false;
    try {
      success = document.execCommand('copy');
    } catch (error) {
      console.error('Failed to copy text using execCommand: ', error);
    }

    // Clean up
    document.body.removeChild(textArea);
    return success;
  };

  return [isCopied, copy];
};

export default useCopyToClipboard;
