// import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';


export default function Message(props: { message: string, isUser: boolean, createdAt: string}) {

    function UserMessage(props: { message: string, createdAt: string}) {
    return (<>
        <span className="flex flex-row">
            <div className="user-message rounded-full text-right text-white px-4">{ props.message }</div>
                {/* <div className={`text-xs text-right text-white-400 font-bold pr-4`}>{ format(new Date(props.createdAt), 'MMM d, yyyy h:mm a') }</div> */}
        </span>
        </>)
    }

    function AssistantMessage(props: { message: string, createdAt: string}) {
    return (<>
        <div className="assistant-message max-w-6xl bg-white-200 rounded-full py-1 px-4 text-left text-gray-600">
            <ReactMarkdown components={{
                code({ node, inline, className, children, ...props }: {
                    node?: object,
                    inline?: boolean;
                    className?: string;
                    children?: React.ReactNode;
                }) {
                    console.log(typeof node);
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                    <SyntaxHighlighter className="flex"
                        language={match[1]}
                        style={dracula}
                        {...props}
                        >
                        {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                ) : (
                    <code className={className} {...props}>
                        {children}
                    </code>
                );
                }
            }}
            >
                { props.message }
            </ReactMarkdown>
        </div>
        {/* <div className={`text-xs text-left text-gray-500 font-bold pl-4`}>{ format(new Date(props.createdAt), 'MMM d, yyyy h:mm a') }</div> */}
    </>)
    }

    return (
        <div className="message flex flex-col">
            { props.isUser ? <UserMessage message={props.message} createdAt={props.createdAt} /> : <AssistantMessage message={props.message} createdAt={props.createdAt} /> }
        </div>
    )
}