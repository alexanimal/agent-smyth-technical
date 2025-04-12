import { useState, useRef, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { addMessage, messageLoading } from '../store/messageSlice';
import { v4 } from 'uuid';

export default function InputContainer() {
    const [inputMessage, setInputMessage] = useState('');
    const [disabled, setDisabled] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const [numResults, setNumResults] = useState(25);
    const [showSettings, setShowSettings] = useState(false);
    const settingsRef = useRef<HTMLDivElement>(null);
    const sliderRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const dispatch = useDispatch();

    // Handle clicking outside settings menu to close it
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
                setShowSettings(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    // Update slider progress on mount and when numResults changes
    useEffect(() => {
        updateSliderProgress();
    }, [numResults]);

    // Auto-focus the input field
    useEffect(() => {
        if (isExpanded && textareaRef.current) {
            textareaRef.current.focus();
            // Set cursor at the end of text
            const length = textareaRef.current.value.length;
            textareaRef.current.setSelectionRange(length, length);
        } else if (!isExpanded && inputRef.current) {
            inputRef.current.focus();
            // Set cursor at the end of text
            const length = inputRef.current.value.length;
            inputRef.current.setSelectionRange(length, length);
        }
    }, [isExpanded]);

    const updateSliderProgress = () => {
        if (sliderRef.current) {
            const min = parseInt(sliderRef.current.min);
            const max = parseInt(sliderRef.current.max);
            const value = numResults;
            const percentage = ((value - min) / (max - min)) * 100;
            sliderRef.current.style.setProperty('--range-progress', `${percentage}%`);
        }
    }

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
        try {
            const resp = await fetch('http://localhost:8003/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': '1234567890'
                },
                body: JSON.stringify({ message, "num_results": numResults })
            });

            if (!resp.ok) {
                throw new Error(`Server responded with status: ${resp.status}`);
            }

            const respJson = await resp.json();
            console.log(respJson);

            dispatch(addMessage({
                message: respJson.response.toString(),
                id: v4(),
                createdAt: new Date().toISOString(),
                isUser: false
            }));
        } catch (error) {
            console.error('Error sending message:', error);
            dispatch(addMessage({
                message: 'Error: Could not connect to the server. Please make sure the backend is running.',
                id: v4(),
                createdAt: new Date().toISOString(),
                isUser: false
            }));
        } finally {
            dispatch(messageLoading(false));
        }
    }

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setInputMessage(e.target.value);
    }

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = Number(e.target.value);
        setNumResults(value);
        // Update progress is now handled by the useEffect
    }

    const toggleExpand = () => {
        setIsExpanded(!isExpanded);
    }

    return (
        <div className="flex items-center justify-center p-4">
            <div className="relative w-full max-w-full">
                { isExpanded ? (
                <div className="flex flex-col w-full">
                    <textarea
                        ref={textareaRef}
                        value={inputMessage}
                        onChange={handleInputChange}
                        className="input-field w-full max-w-5xl px-4 py-3 h-24 rounded-lg border-2 border-transparent focus:border-blue-500 bg-gray-800 text-white placeholder:text-gray-400 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-600 transition-all duration-300"
                        placeholder="Type your message here..."
                        disabled={disabled}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSendMessage(inputMessage);
                            }
                        }}
                    />
                    <div className="text-xs text-gray-400 mt-1 ml-2">
                        Press Shift+Enter for new line, Enter to send
                    </div>
                </div>
            ) : (
                <input
                    ref={inputRef}
                    type="text"
                    value={inputMessage}
                    onChange={handleInputChange}
                    className="input-field w-full max-w-3xl px-4 py-3 rounded-full border-2 border-transparent focus:border-blue-500 bg-gray-800 text-white placeholder:text-gray-400 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-600 transition-all duration-300"
                    placeholder="Type your message here..."
                    disabled={disabled}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            handleSendMessage(inputMessage);
                        }
                    }}
                />
            )}
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex space-x-2">
                <button
                    type="button"
                    onClick={toggleExpand}
                    className="flex items-center justify-center bg-gray-700 text-white rounded-full h-10 w-10 shadow-md hover:shadow-lg transition-all duration-300 ease-in-out hover:bg-gray-600 mr-2"
                    aria-label={isExpanded ? "Collapse" : "Expand"}
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        {isExpanded ? (
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                        ) : (
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7"></path>
                        )}
                    </svg>
                </button>

                <div ref={settingsRef} className="relative">
                    <button
                        type="button"
                        onClick={() => setShowSettings(!showSettings)}
                        className="flex items-center justify-center bg-gray-700 text-white rounded-full h-10 w-10 shadow-md hover:shadow-lg transition-all duration-300 ease-in-out hover:bg-gray-600 mr-2"
                        aria-label="Settings"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                        </svg>
                    </button>

                    {showSettings && (
                        <div className="absolute bottom-14 right-0 w-72 bg-gray-800 border border-gray-700 shadow-xl rounded-lg p-4 z-50">
                            <h3 className="text-white font-medium mb-2">Settings</h3>
                            <div className="mb-3">
                                <div className="flex justify-between items-center mb-2">
                                    <label className="block text-sm font-medium text-gray-300">
                                        Number of Results:
                                    </label>
                                    <span className="text-white font-medium bg-gray-700 px-2 py-1 rounded-md">
                                        {numResults}
                                    </span>
                                </div>

                                <div className="relative pb-6">
                                    <input
                                        ref={sliderRef}
                                        type="range"
                                        min="5"
                                        max="250"
                                        value={numResults}
                                        onChange={handleSliderChange}
                                        className="w-full"
                                    />
                                    <div className="absolute left-0 right-0 -bottom-1 flex justify-between text-xs text-gray-400">
                                        <span>5</span>
                                        <span>250</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

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
