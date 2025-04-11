export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
  sources?: string[];
  alternativeViewpoint?: string | null;
}

export interface ChatStore {
  messages: ChatMessage[];
  streamingMessage: string | null;
  isLoading: boolean;

  addMessage: (message: ChatMessage) => void;
  setStreamingMessage: (content: string | null) => void;
  getStreamingMessage: () => string | null;
  finalizeStreamingMessage: () => void;
  setLoading: (isLoading: boolean) => void;
  clearMessages: () => void;
}

export interface SettingsStore {
  model: string;
  apiType: 'regular' | 'stream';
  generateAlternativeOpinions: boolean;
  numRecords: number;

  setModel: (model: string) => void;
  setApiType: (apiType: 'regular' | 'stream') => void;
  setGenerateAlternativeOpinions: (value: boolean) => void;
  setNumRecords: (value: number) => void;
}

export interface SendMessageParams {
  message: string;
  settings: {
    model: string;
    apiType: 'regular' | 'stream';
    generateAlternativeOpinions: boolean;
    numRecords: number;
  };
  abortController?: AbortController;
}
