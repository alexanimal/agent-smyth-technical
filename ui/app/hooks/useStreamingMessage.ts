import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useChatStore } from '../store/chatStore';
import { StreamProcessResult } from '../utils/streamProcessor';

/**
 * Custom hook for managing streaming message state and finalization with metadata
 */
export function useStreamingMessage() {
  const {
    setStreamingMessage,
    addMessage,
    finalizeStreamingMessage: originalFinalize,
    getStreamingMessage
  } = useChatStore();

  // Local state for tracking metadata received during streaming
  const [pendingMetadata, setPendingMetadata] = useState<any>(null);
  const [pendingSources, setPendingSources] = useState<string[]>([]);
  const [pendingAlternativeViewpoint, setPendingAlternativeViewpoint] = useState<string | undefined>(undefined);
  const [isStreamComplete, setIsStreamComplete] = useState<boolean>(false);
  const [streamCompleteData, setStreamCompleteData] = useState<StreamProcessResult | null>(null);

  /**
   * Update the streaming message content
   * @param content The new content to display
   */
  const updateStreamingContent = useCallback((content: string) => {
    setStreamingMessage(content);
  }, [setStreamingMessage]);

  /**
   * Store streaming metadata for use when finalizing the message
   * @param metadata The metadata object
   */
  const setStreamingMetadata = useCallback((metadata: any) => {
    setPendingMetadata(metadata);
    console.log('Stored pending metadata:', metadata);
  }, []);

  /**
   * Store streaming sources for use when finalizing the message
   * @param sources Array of source strings
   */
  const setStreamingSources = useCallback((sources: string[]) => {
    if (Array.isArray(sources)) {
      setPendingSources(sources);
      console.log('Stored sources:', sources);
    } else {
      console.warn('Attempted to store sources but received non-array value:', sources);
    }
  }, []);

  /**
   * Store alternative viewpoint for use when finalizing the message
   * @param viewpoint Alternative viewpoint text
   */
  const setStreamingAlternativeViewpoint = useCallback((viewpoint?: string) => {
    setPendingAlternativeViewpoint(viewpoint);
    console.log('Stored alternative viewpoint:', viewpoint ? viewpoint.substring(0, 50) + '...' : undefined);
  }, []);

  /**
   * Handle the completion of a stream with all necessary metadata
   * @param result The complete stream processing result with metadata
   */
  const handleStreamComplete = useCallback((result: StreamProcessResult) => {
    console.log('Stream complete handler called with:', {
      contentLength: result.content.length,
      hasSources: Array.isArray(result.sources) && result.sources.length > 0,
      sourcesCount: Array.isArray(result.sources) ? result.sources.length : 0,
      hasAlternativeViewpoint: !!result.alternativeViewpoint,
      alternativeViewpointLength: result.alternativeViewpoint?.length || 0,
      hasMetadata: !!result.metadata
    });

    // Store the complete result data for potential later use
    setStreamCompleteData(result);
    setIsStreamComplete(true);

    // Store all metadata from the complete event
    if (result.metadata) {
      setStreamingMetadata(result.metadata);
    }

    if (result.sources && Array.isArray(result.sources) && result.sources.length > 0) {
      setStreamingSources(result.sources);
    }

    if (result.alternativeViewpoint) {
      setStreamingAlternativeViewpoint(result.alternativeViewpoint);
    }

    // Update the streaming content with the complete content
    updateStreamingContent(result.content);

    console.log('Stream completed with metadata', {
      hasSources: Array.isArray(result.sources) && result.sources.length > 0,
      sourcesCount: Array.isArray(result.sources) ? result.sources.length : 0,
      hasAlternativeViewpoint: !!result.alternativeViewpoint,
      alternativeViewpointPreview: result.alternativeViewpoint
        ? result.alternativeViewpoint.substring(0, 50) + '...'
        : undefined,
      hasMetadata: !!result.metadata,
      content: result.content.substring(0, 50) + '...'
    });

    // Finalize the message automatically if there's content
    if (result.content.trim()) {
      // Small delay to ensure UI has updated with the final content
      setTimeout(() => {
        finalizeStreamingMessageWithMetadata(result);
      }, 300); // Increased delay to ensure UI has time to update
    }
  }, [updateStreamingContent, setStreamingMetadata, setStreamingSources, setStreamingAlternativeViewpoint]);

  /**
   * Finalize a streaming message with all collected metadata
   * @param completeResult Optional complete result to use for finalization
   */
  const finalizeStreamingMessageWithMetadata = useCallback((completeResult?: StreamProcessResult) => {
    // If we have a complete result passed in, use that data directly
    const dataToUse = completeResult || streamCompleteData;

    // If there's no streaming message from the store's perspective, don't do anything
    const messageContent = dataToUse?.content || getStreamingMessage();
    if (!messageContent) {
      console.warn('No streaming message content found to finalize');
      return;
    }

    // Use sources from the result or from state
    const sourcesToUse = (dataToUse?.sources && Array.isArray(dataToUse.sources) && dataToUse.sources.length > 0)
      ? dataToUse.sources
      : pendingSources;

    // Use alternative viewpoint from the result or from state
    const altViewpointToUse = dataToUse?.alternativeViewpoint || pendingAlternativeViewpoint;

    console.log('Finalizing message with metadata', {
      contentLength: messageContent.length,
      hasSources: Array.isArray(sourcesToUse) && sourcesToUse.length > 0,
      sourcesCount: Array.isArray(sourcesToUse) ? sourcesToUse.length : 0,
      hasAlternativeViewpoint: !!altViewpointToUse,
      alternativeViewpointPreview: altViewpointToUse
        ? altViewpointToUse.substring(0, 50) + '...'
        : undefined,
      hasMetadata: !!(dataToUse?.metadata || pendingMetadata)
    });

    // Add the message to the chat with metadata
    addMessage({
      id: uuidv4(),
      role: 'assistant',
      content: messageContent,
      timestamp: Date.now(),
      sources: sourcesToUse,
      alternativeViewpoint: altViewpointToUse
    });

    // Clear streaming state
    resetStreaming();
  }, [
    addMessage,
    pendingMetadata,
    pendingSources,
    pendingAlternativeViewpoint,
    setStreamingMessage,
    getStreamingMessage,
    streamCompleteData
  ]);

  /**
   * Reset the streaming state (used for cancellation or errors)
   */
  const resetStreaming = useCallback(() => {
    setStreamingMessage(null);
    setPendingMetadata(null);
    setPendingSources([]);
    setPendingAlternativeViewpoint(undefined);
    setIsStreamComplete(false);
    setStreamCompleteData(null);
  }, [setStreamingMessage]);

  return {
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
  };
}
