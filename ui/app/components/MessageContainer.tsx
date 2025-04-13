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
        <div className="flex justify-end mb-4">
            <div className="message-container w-fit shadow-lg shadow-white-700/50 p-4 bg-blue-700/50 text-left rounded-lg hover:bg-blue-500 transition-all duration-300 max-w-[70%] md:max-w-[60%]">
                <Message message={props.message} isUser={props.isUser} createdAt={props.createdAt} />
                <div ref={messagesEndRef} />
            </div>
        </div> :
        <div className="flex justify-start mb-4">
            <Avatar />
            <div className="message-container w-fit ml-3 shadow-lg shadow-white-700/50 p-4 bg-gray-100 text-left rounded-lg hover:bg-gray-200 transition-all duration-300 max-w-[70%] md:max-w-[70%]">
                <Message message={props.message} isUser={props.isUser} createdAt={props.createdAt} />
                <div ref={messagesEndRef} />
            </div>
        </div>
    )
}
