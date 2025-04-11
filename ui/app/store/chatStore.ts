import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { v4 as uuidv4 } from 'uuid'
import { ChatStore, ChatMessage } from './types'

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      messages: [],
      streamingMessage: null,
      isLoading: false,

      addMessage: (message: ChatMessage) =>
        set((state) => ({
          messages: [...state.messages, message]
        })),

      setStreamingMessage: (content: string | null) =>
        set(() => ({
          streamingMessage: content
        })),

      getStreamingMessage: () => get().streamingMessage,

      finalizeStreamingMessage: () =>
        set((state) => {
          if (!state.streamingMessage) return state

          const newMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: state.streamingMessage,
            timestamp: Date.now()
          }

          return {
            messages: [...state.messages, newMessage],
            streamingMessage: null
          }
        }),

      setLoading: (isLoading: boolean) =>
        set(() => ({
          isLoading: isLoading
        })),

      clearMessages: () =>
        set(() => ({
          messages: [],
          streamingMessage: null
        })),
    }),
    {
      name: 'chat-storage',
      storage: createJSONStorage(() => {
        if (typeof window === 'undefined') {
          return {
            getItem: () => null,
            setItem: () => {},
            removeItem: () => {}
          }
        }
        return localStorage
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          console.log('Chat storage rehydrated')
        } else {
          console.error('Failed to rehydrate chat storage')
        }
      },
      partialize: (state) => ({
        messages: state.messages
      }),
      version: 1
    }
  )
)
