// import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';


export default function Message(props: { message: string, isUser: boolean, createdAt: string}) {

    function UserMessage(props: { message: string, createdAt: string}) {
    return (<>
        <span className="flex flex-row">
            <div className="user-message w-fit rounded-md text-white px-4 py-2 text-left break-words whitespace-pre-wrap">{ props.message }</div>
                {/* <div className={`text-xs text-right text-white-400 font-bold pr-4`}>{ format(new Date(props.createdAt), 'MMM d, yyyy h:mm a') }</div> */}
        </span>
        </>)
    }

    function AssistantMessage(props: { message: string, createdAt: string}) {
    return (<>
        <div className="assistant-message w-fit py-1 px-2 text-left text-gray-600 break-words whitespace-pre-wrap">
            <ReactMarkdown components={{
                code({ node, inline, className, children, ...props }: {
                    node?: object,
                    inline?: boolean;
                    className?: string;
                    children?: React.ReactNode;
                }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                        <div className="code-block-wrapper my-2 rounded overflow-hidden">
                            <div className="code-header bg-gray-800 text-gray-200 text-xs px-4 py-1 flex justify-between items-center">
                                <span>{match[1]}</span>
                            </div>
                            <SyntaxHighlighter
                                language={match[1]}
                                style={dracula}
                                customStyle={{ margin: 0, borderRadius: 0 }}
                                {...props}
                            >
                                {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                        </div>
                    ) : (
                        <code className={`${className} bg-gray-100 px-1 py-0.5 rounded text-sm`} {...props}>
                            {children}
                        </code>
                    );
                },
                p: ({ children }) => <p className="my-2 text-sm leading-relaxed text-left">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-5 my-2 space-y-1 text-left">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 my-2 space-y-1 text-left">{children}</ol>,
                li: ({ children }) => <li className="my-1 text-sm text-left">{children}</li>,
                blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-gray-300 pl-3 italic my-3 text-gray-700 text-left">{children}</blockquote>
                ),
                a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-600 hover:underline">
                        {children}
                    </a>
                )
            }}
            >
                { props.message }
            </ReactMarkdown>
        </div>
        {/* <div className={`text-xs text-left text-gray-500 font-bold pl-4`}>{ format(new Date(props.createdAt), 'MMM d, yyyy h:mm a') }</div> */}
    </>)
    }

    return (
        <div className="message text-left w-fit">
            { props.isUser ? <UserMessage message={props.message} createdAt={props.createdAt} /> : <AssistantMessage message={props.message} createdAt={props.createdAt} /> }
        </div>
    )
}
