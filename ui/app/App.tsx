
import './App.css'
import { Suspense } from 'react'
import ChatContainer from './components/ChatContainer'
function App() {
  
  // Fix: prefetch and preconnect are not callable functions from react-dom
  // They should be used as resource hints in the document head
  
  return (
    <>
      <Suspense fallback={<div>Loading...</div>}>
        <ChatContainer />
      </Suspense>
    </>
  )
}

export default App
