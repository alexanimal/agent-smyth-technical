import React, { useEffect } from 'react'
import { useSettingsStore } from '../../store/settingsStore'
import { ModelSelector } from './ModelSelector'
import { APISelector } from './APISelector'
import { FeatureToggle } from './FeatureToggle'
import { Slider } from './Slider'

interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
}

/**
 * Modal panel for adjusting chat settings
 * Combines all settings components and handles backdrop clicks
 */
export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const settings = useSettingsStore()

  // Disable scrolling on the body when the modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }

    return () => {
      document.body.style.overflow = ''
    }
  }, [isOpen])

  // Handle ESC key to close the modal
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }

    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [isOpen, onClose])

  if (!isOpen) return null

  // Close when clicking outside the modal
  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 animate-fadeIn"
      onClick={handleBackdropClick}
      data-testid="settings-panel"
    >
      <div
        className="bg-white dark:bg-gray-800 dark:text-white rounded-lg w-[500px] max-w-full p-6 shadow-xl animate-scaleIn"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-bold">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl"
            aria-label="Close settings"
            data-testid="close-settings"
          >
            &times;
          </button>
        </div>

        <div className="space-y-8">
          <ModelSelector
            value={settings.model}
            onChange={settings.setModel}
          />

          <APISelector
            value={settings.apiType}
            onChange={settings.setApiType}
          />

          <Slider
            label="Number of Records"
            min={5}
            max={250}
            step={5}
            value={settings.numRecords}
            onChange={settings.setNumRecords}
            description="Set the maximum number of records to process (5-250)"
            testId="num-records-slider"
          />

          <FeatureToggle
            label="Generate Alternative Opinions"
            value={settings.generateAlternativeOpinions}
            onChange={settings.setGenerateAlternativeOpinions}
            description="AI will try to provide different perspectives on topics when appropriate."
            testId="alternative-opinions-toggle"
          />
        </div>

        <div className="mt-8 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            data-testid="save-settings"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  )
}
