import React from 'react'

/**
 * Available AI models for the chat
 */
const AVAILABLE_MODELS = [
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
  { id: 'gpt-4', name: 'GPT-4' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
]

interface ModelSelectorProps {
  value: string
  onChange: (model: string) => void
}

/**
 * Component for selecting an AI model from available options
 */
export const ModelSelector: React.FC<ModelSelectorProps> = ({ value, onChange }) => {
  return (
    <div className="space-y-2">
      <label htmlFor="modelSelect" className="block text-sm font-medium">
        AI Model
      </label>
      <select
        id="modelSelect"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500
          dark:bg-gray-800 dark:text-white dark:border-gray-700"
        data-testid="model-selector"
      >
        {AVAILABLE_MODELS.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
    </div>
  )
}
