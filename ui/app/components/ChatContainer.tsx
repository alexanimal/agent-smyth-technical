'use client';

import { useSelector } from 'react-redux';
import { useRef, useEffect } from 'react';

import MessageContainer from './MessageContainer';
import { RootState } from '../store/store';
import InputContainer from './InputContainer';
import Loader from './Loader';

// Define an interface for the chat message structure
interface ChatMessage {
  id: string | number; // Or the actual type of your ID
  message: string;
  isUser: boolean;
  createdAt: string;
}

export default function ChatContainer() {
    const chats = useSelector((state: RootState) => state.messages);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };
    
    useEffect(() => {
        scrollToBottom();
    }, [chats.messages]);

    return (
        <main className="chat-container flex flex-col h-screen" >
            <div className="chat-container-messages flex overflow-y-auto scrollbar-none p-4">
                <div className="chat-container-messages-content flex flex-col gap-6 w-full max-w-full mx-auto mt-auto">
                    {chats.messages.map((m: ChatMessage) => (
                        <MessageContainer 
                            key={m.id} 
                            message={m.message} 
                            isUser={m.isUser} 
                            createdAt={m.createdAt} 
                        />
                    ))}
                    <div ref={messagesEndRef} />
                    <Loader />
                </div>
            </div>
            <InputContainer />
        </main>
    )
}