import { useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { sendMessage } from '../api/chatAPI';
import { processStreamWithMetadata } from '../utils/streamProcessor';
import { useStreamingMessage } from './useStreamingMessage';
import { ChatMessage } from '../store/types';

/**
 * Enhanced chat hook that properly handles metadata in both streaming and non-streaming responses
 */
export const useChatWithMetadata = () => {
  const {
    messages,
    addMessage,
    setLoading,
    isLoading,
  } = useChatStore();

  // Get streaming utilities from our custom hook
  const {
    updateStreamingContent,
    setStreamingMetadata,
    setStreamingSources,
    setStreamingAlternativeViewpoint,
    handleStreamComplete,
    finalizeStreamingMessageWithMetadata,
    resetStreaming,
    pendingMetadata,
    pendingSources,
    pendingAlternativeViewpoint,
    isStreamComplete
  } = useStreamingMessage();

  const settings = useSettingsStore();
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastStreamResultRef = useRef<any>(null);

  // Debug effect to monitor changes in streaming metadata
  useEffect(() => {
    if (isStreamComplete) {
      console.log('Stream completed status changed with metadata:', {
        hasSources: Array.isArray(pendingSources) && pendingSources.length > 0,
        sourcesCount: pendingSources.length,
        hasAlternativeViewpoint: !!pendingAlternativeViewpoint,
        metadataAvailable: !!pendingMetadata
      });
    }
  }, [isStreamComplete, pendingSources, pendingAlternativeViewpoint, pendingMetadata]);

  /**
   * Sends a user message to the chat API
   * Handles both streaming and regular responses with proper metadata handling
   */
  const sendUserMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    // Reset last stream result
    lastStreamResultRef.current = null;

    // Create user message
    const userMessage: ChatMessage = {
      role: 'user',
      content,
      id: uuidv4(),
      timestamp: Date.now()
    };

    addMessage(userMessage);
    setLoading(true);

    try {
      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController();

      console.log('Sending message to server:', content);
      const response = await sendMessage({
        message: content,
        settings,
        abortController: abortControllerRef.current
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      if (settings.apiType === 'stream') {
        // Handle streaming response with metadata
        console.log('Processing streaming response with metadata support');

        try {
          // Process the stream and update UI as chunks arrive, and handle the complete event
          const result = await processStreamWithMetadata(
            response,
            (incrementalContent) => {
              // Update UI with each chunk
              updateStreamingContent(incrementalContent);
            },
            (completeResult) => {
              // This callback is called when the complete event is received
              console.log('Stream complete event received:', {
                contentLength: completeResult.content.length,
                hasSources: Array.isArray(completeResult.sources) && completeResult.sources.length > 0,
                sourcesCount: completeResult.sources?.length || 0,
                hasAlternativeViewpoint: !!completeResult.alternativeViewpoint,
                hasMetadata: !!completeResult.metadata,
                isComplete: completeResult.isComplete
              });

              // Store the result for potential use later
              lastStreamResultRef.current = completeResult;

              // Handle the complete event with all metadata
              handleStreamComplete(completeResult);
            }
          );

          console.log('Stream processing finished with result:', {
            contentLength: result.content.length,
            hasSources: Array.isArray(result.sources) && result.sources.length > 0,
            sourcesCount: result.sources?.length || 0,
            hasAlternativeViewpoint: !!result.alternativeViewpoint,
            hasMetadata: !!result.metadata,
            isComplete: result.isComplete
          });

          // In case the handleStreamComplete callback didn't work properly,
          // we'll ensure we finalize the message here as a fallback
          if (isStreamComplete && lastStreamResultRef.current) {
            finalizeStreamingMessageWithMetadata(lastStreamResultRef.current);
          }
        } catch (streamError) {
          console.error('Error processing stream:', streamError);
          resetStreaming();
          addMessage({
            role: 'system',
            content: `Error processing stream: ${streamError}`,
            id: uuidv4(),
            timestamp: Date.now()
          });
        }
      } else {
        // Handle regular response
        console.log('Processing regular response');
        try {
          const data = await response.json();
          console.log('Received regular response data:', {
            responseLength: data.response?.length || 0,
            hasSources: Array.isArray(data.sources) && data.sources.length > 0,
            sourcesCount: data.sources?.length || 0,
            hasAlternativeViewpoints: !!data.alternative_viewpoints
          });

          addMessage({
            role: 'assistant',
            content: data.response || data.content || 'No content received',
            id: uuidv4(),
            timestamp: Date.now(),
            sources: Array.isArray(data.sources) ? data.sources : [],
            alternativeViewpoint: data.alternative_viewpoints || null
          });
        } catch (jsonError) {
          console.error('Error parsing JSON response:', jsonError);
          addMessage({
            role: 'system',
            content: `Error parsing response: ${jsonError}`,
            id: uuidv4(),
            timestamp: Date.now()
          });
        }
      }
    } catch (error: any) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        console.error('Error sending message:', error);
        addMessage({
          role: 'system',
          content: `Error: ${error.message || 'Failed to send message. Please try again.'}`,
          id: uuidv4(),
          timestamp: Date.now()
        });
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  }, [
    addMessage,
    setLoading,
    settings,
    updateStreamingContent,
    handleStreamComplete,
    finalizeStreamingMessageWithMetadata,
    resetStreaming,
    isStreamComplete
  ]);

  /**
   * Cancels the current request if one is in progress
   */
  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);
      resetStreaming();
    }
  }, [setLoading, resetStreaming]);

  return {
    messages,
    sendMessage: sendUserMessage,
    cancelRequest,
    isLoading,
    streamingMetadata: pendingMetadata,
    streamingSources: pendingSources,
    streamingAlternativeViewpoint: pendingAlternativeViewpoint,
    isStreamComplete
  };
};

export default useChatWithMetadata;
