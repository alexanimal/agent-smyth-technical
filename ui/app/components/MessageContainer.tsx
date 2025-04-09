import Message from './Message';
import Avatar from './Avatar';
import { useRef, useEffect } from 'react';

export default function MessageContainer(props: { message: string, isUser: boolean, createdAt: string }) {
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, []);

    return (
        props.isUser ? 
        <div className="flex justify-end">
            <div className="message-container flex flex-col max-w-6xl shadow-lg shadow-white-700/50 p-4 bg-blue-700/50 rounded-full hover:bg-blue-500 transition-all duration-300">
                <Message message={props.message} isUser={props.isUser} createdAt={props.createdAt} />
                <div ref={messagesEndRef} />
            </div>
        </div> : <div className="flex justify-start">
            <Avatar />
            <div className="message-container flex flex-col max-w-6xl shadow-lg shadow-white-700/50 p-4 bg-gray-200 rounded-md hover:bg-gray-200 transition-all duration-300">
                <Message message={props.message} isUser={props.isUser} createdAt={props.createdAt} />
                <div ref={messagesEndRef} />
            </div>
        </div>
    )
}