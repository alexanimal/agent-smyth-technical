import React, { useState, useEffect } from 'react'

interface SliderProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
  description?: string;
  testId?: string;
}

/**
 * Reusable slider component with label and value display
 */
export const Slider: React.FC<SliderProps> = ({
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
  description,
  testId = 'slider'
}) => {
  const [localValue, setLocalValue] = useState<number>(value)

  // Update local value when prop value changes
  useEffect(() => {
    setLocalValue(value)
  }, [value])

  // Handle slider change
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10)
    setLocalValue(newValue)
  }

  // Handle final value change when slider is released
  const handleSliderChangeComplete = () => {
    onChange(localValue)
  }

  // Handle direct input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10)
    if (!isNaN(newValue)) {
      const constrainedValue = Math.max(min, Math.min(max, newValue))
      setLocalValue(constrainedValue)
      onChange(constrainedValue)
    }
  }

  // Calculate the percentage for the track fill
  const percentage = ((localValue - min) / (max - min)) * 100

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="block text-sm font-medium">
          {label}
        </label>
        <div className="flex gap-2 items-center">
          <input
            type="number"
            min={min}
            max={max}
            value={localValue}
            onChange={handleInputChange}
            className="w-16 py-1 px-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500
              dark:bg-gray-800 dark:text-white dark:border-gray-700"
            data-testid={`${testId}-input`}
          />
          <span className="text-sm text-gray-600 dark:text-gray-400">{localValue}</span>
        </div>
      </div>

      <div className="relative h-6 flex items-center">
        <div className="absolute w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500"
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleSliderChange}
          onMouseUp={handleSliderChangeComplete}
          onTouchEnd={handleSliderChangeComplete}
          className="absolute w-full h-2 opacity-0 cursor-pointer"
          data-testid={testId}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 px-2">
        <span>{min}</span>
        <span>{max}</span>
      </div>

      {description && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {description}
        </p>
      )}
    </div>
  )
}
