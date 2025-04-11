import { useState, useEffect } from 'react';

/**
 * Custom hook for interacting with localStorage
 * Handles SSR, errors, and provides a convenient interface
 *
 * @param key - The localStorage key to use
 * @param initialValue - The initial value if no value exists in localStorage
 */
export const useLocalStorage = <T>(key: string, initialValue: T): [T, (value: T | ((val: T) => T)) => void] => {
  // Get from localStorage on initial render, falling back to initialValue
  const [storedValue, setStoredValue] = useState<T>(() => {
    // Check if running in browser environment
    if (typeof window === 'undefined') {
      return initialValue;
    }

    try {
      // Get from localStorage
      const item = window.localStorage.getItem(key);
      // Parse stored json, or return initialValue if no value
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  // Update localStorage when the state changes
  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      // Save state to localStorage
      window.localStorage.setItem(key, JSON.stringify(storedValue));
    } catch (error) {
      console.error(`Error saving to localStorage key "${key}":`, error);

      // If error is quota exceeded, try to handle it gracefully
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        console.warn('localStorage quota exceeded. Some data may not be saved.');
        // Could implement fallback behavior here
      }
    }
  }, [key, storedValue]);

  // Return a wrapped setter that updates state
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      // Allow value to be a function for same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;

      // Save state
      setStoredValue(valueToStore);
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  };

  return [storedValue, setValue];
};

export default useLocalStorage;
