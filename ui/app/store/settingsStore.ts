import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { SettingsStore } from './types'

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      model: 'gpt-3.5-turbo',
      apiType: 'stream',
      generateAlternativeOpinions: false,
      numRecords: 25,

      setModel: (model: string) =>
        set(() => ({ model })),

      setApiType: (apiType: 'regular' | 'stream') =>
        set(() => ({ apiType })),

      setGenerateAlternativeOpinions: (value: boolean) =>
        set(() => ({ generateAlternativeOpinions: value })),

      setNumRecords: (value: number) =>
        set(() => ({ numRecords: value }))
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => {
        // Add SSR safety check
        if (typeof window === 'undefined') {
          return {
            getItem: () => null,
            setItem: () => {},
            removeItem: () => {}
          }
        }
        return localStorage
      }),
      version: 1
    }
  )
)
