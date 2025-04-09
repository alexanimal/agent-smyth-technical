import { useSelector } from 'react-redux';
import { RootState } from '../store/store';

export default function Loader() {
    const isLoading = useSelector((state: RootState) => state.messages.loading);
    console.log(isLoading);

    return (
        <div className="flex space-x-2 " style={{ display: isLoading ? 'flex' : 'none' }}>
            <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0s" }}></div>
            <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
            <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></div>
        </div>
    )
}