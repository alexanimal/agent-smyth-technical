import './App.css'
import { Suspense } from 'react'
import ChatContainer from './components/ChatContainer'

/**
 * Root App component that wraps the chat application
 * Uses Suspense for better loading experience
 */
function App() {
  return (
    <div className="h-screen w-screen">
      <Suspense fallback={
        <div className="flex h-screen w-screen items-center justify-center">
          <div className="text-center">
            <div className="mb-4 text-2xl font-semibold">Loading Chat...</div>
            <div className="h-2 w-40 mx-auto rounded-full bg-gray-200 overflow-hidden">
              <div className="h-full bg-blue-500 animate-pulse"></div>
            </div>
          </div>
        </div>
      }>
        <ChatContainer />
      </Suspense>
    </div>
  )
}

export default App
