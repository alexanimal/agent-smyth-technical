import { useState } from 'react';
import { useDispatch } from 'react-redux';
import { addMessage, messageLoading } from '../store/messageSlice';
import { v4 } from 'uuid';

export default function InputContainer() {
    const [inputMessage, setInputMessage] = useState('');
    const [disabled, setDisabled] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const dispatch = useDispatch();

    const handleSendMessage = (message: string) => {
        if (!message.trim()) return;
        setDisabled(true);
        dispatch(addMessage({ message, id: v4(), createdAt: new Date().toISOString(), isUser: true }));
        
        sendMessageToAgent(message);
        setInputMessage('');
        setDisabled(false);
        setIsExpanded(false);
    }

    const sendMessageToAgent = async (message: string) => {
        dispatch(messageLoading(true));
        const resp = await fetch('http://localhost:8003/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': '1234567890'
            },
            body: JSON.stringify({ message })
        })
        const respJson = await resp.json();
        console.log(respJson);

        dispatch(addMessage({ message: respJson.response.toString(), id: v4(), createdAt: new Date().toISOString(), isUser: false }));
        dispatch(messageLoading(false));
    }

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setInputMessage(e.target.value);
    }
    
    return (

        <div className="flex items-center justify-center p-4">
            <div className="relative w-full max-w-full">
                { isExpanded ? (
                <textarea
                    value={inputMessage}
                    onChange={handleInputChange}
                    className="input-field w-full max-w-5xl px-4 py-2 h-24 rounded-sm border-2 border-transparent focus:border-blue-500 bg-gray-800 text-white placeholder:text-gray-400 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-600 hover:scale-105 duration-300 ease-in-out"
                    placeholder="Type your message here..."
                    disabled={disabled}
                />
            ) : (
                <input
                type={'text'}
                value={inputMessage}
                onChange={handleInputChange}
                className="input-field w-full max-w-5xl px-4 py-3 rounded-full border-2 border-transparent focus:border-blue-500 bg-gray-800 text-white placeholder:text-gray-400 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-600 hover:scale-105 duration-300 ease-in-out"
                placeholder="Type your message here..."
                disabled={disabled}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                        handleSendMessage(inputMessage);
                    }
                }}
            />)}
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex space-x-2">

                <button type="button" onClick={() => handleSendMessage(inputMessage)} disabled={disabled} className="flex items-center justify-center bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-full h-14 w-14 shadow-lg hover:shadow-xl transition-all duration-300 ease-in-out hover:scale-105">
                    <svg className="w-4 h-4" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 10">
                        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M1 5h12m0 0L9 1m4 4L9 9"/>
                    </svg>
                </button>
            </div>
            </div>
        </div>
    )
}
