import React from 'react'

interface APISelectorProps {
  value: 'regular' | 'stream'
  onChange: (type: 'regular' | 'stream') => void
}

/**
 * Component for selecting the API response type (regular or streaming)
 */
export const APISelector: React.FC<APISelectorProps> = ({ value, onChange }) => {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">API Response Type</span>
      <div className="flex gap-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="radio"
            checked={value === 'regular'}
            onChange={() => onChange('regular')}
            className="w-4 h-4 accent-blue-500"
            data-testid="api-type-regular"
          />
          <span>Regular</span>
        </label>

        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="radio"
            checked={value === 'stream'}
            onChange={() => onChange('stream')}
            className="w-4 h-4 accent-blue-500"
            data-testid="api-type-stream"
          />
          <span>Streaming</span>
        </label>
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
        {value === 'stream'
          ? 'See responses as they are generated in real-time.'
          : 'Wait for the complete response before displaying it.'}
      </p>
    </div>
  )
}
