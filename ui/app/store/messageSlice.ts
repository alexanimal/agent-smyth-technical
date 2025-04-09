import { createSlice } from "@reduxjs/toolkit";

export interface Message {
    id: string,
    message: string,
    createdAt: string,
    isUser: boolean
}

export interface Chats {
    messages: Message[],
    loading: boolean
}

const initialState: Chats = {
    messages: [
        {
            id: '1',
            message: 'Ask me about stocks and I will tell you whether you should buy or sell. I can also generate comprehensive trading thesis for any stock.',
            createdAt: new Date().toISOString(),
            isUser: false
        }
    ],
    loading: false
}

const messageSlice = createSlice({
    name: 'messages',
    initialState,
    reducers: {
        addMessage: (state, action) => {
            state.messages.push(action.payload)
        },
        messageLoading: (state, action) => {
            state.loading = action.payload
        }
    }
})

export const { addMessage, messageLoading } = messageSlice.actions
export default messageSlice.reducer