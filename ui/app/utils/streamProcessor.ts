import { processStreamResponse } from '../api/chatAPI';

/**
 * Interface for complete response data that includes metadata
 */
interface CompleteResponseData {
  response: string;
  sources?: string[];
  alternative_viewpoints?: string;
  metadata?: any;
  request_id?: string;
  processing_time?: number;
  [key: string]: any;
}

/**
 * Interface for incremental chunk response
 */
interface ChunkResponseData {
  text: string;
  chunk_id: number;
}

/**
 * Structure to hold processed stream result data
 */
export interface StreamProcessResult {
  content: string;
  isComplete: boolean;
  sources?: string[];
  alternativeViewpoint?: string;
  metadata?: any;
  request_id?: string;
}

/**
 * Process a streaming response and extract both content and metadata
 * @param response The response from the fetch API
 * @param onChunkReceived Callback function called when a chunk is received (for updating UI)
 * @param onCompleteReceived Optional callback for when the complete event is received (with full metadata)
 * @returns Object containing the final content and any metadata
 */
export async function processStreamWithMetadata(
  response: Response,
  onChunkReceived?: (content: string) => void,
  onCompleteReceived?: (result: StreamProcessResult) => void
): Promise<StreamProcessResult> {
  let fullContent = '';
  let sources: string[] = [];
  let alternativeViewpoint: string | undefined = undefined;
  let metadata: any = undefined;
  let requestId: string | undefined = undefined;
  let isComplete = false;

  try {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    const processEvents = async () => {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          // Handle any remaining buffer data
          if (buffer.trim()) {
            processEventData(buffer);
          }
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete events
        let eventEndIndex;
        while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
          const eventData = buffer.substring(0, eventEndIndex);
          buffer = buffer.substring(eventEndIndex + 2);

          processEventData(eventData);
        }
      }
    };

    const processEventData = (eventData: string) => {
      const lines = eventData.split('\n');
      let eventType = '';
      let data = '';

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.substring(6).trim();
        } else if (line.startsWith('data:')) {
          data = line.substring(5).trim();
        }
      }

      if (!eventType || !data) return;

      try {
        const parsedData = JSON.parse(data);

        if (eventType === 'chunk') {
          const chunkData = parsedData as ChunkResponseData;
          fullContent += chunkData.text;
          onChunkReceived?.(fullContent);
        }
        else if (eventType === 'complete') {
          const completeData = parsedData as CompleteResponseData;

          console.log('Received complete event data:', JSON.stringify({
            hasResponse: !!completeData.response,
            responseLength: completeData.response?.length || 0,
            hasSources: Array.isArray(completeData.sources) && completeData.sources.length > 0,
            sourcesCount: Array.isArray(completeData.sources) ? completeData.sources.length : 0,
            hasAlternativeViewpoints: !!completeData.alternative_viewpoints,
            hasMetadata: !!completeData.metadata,
            requestId: completeData.request_id
          }));

          // Use the full response from the complete event
          if (completeData.response) {
            fullContent = completeData.response;
          }

          // Extract metadata with proper type checking
          if (Array.isArray(completeData.sources)) {
            sources = completeData.sources;
          }

          // Properly handle the alternative_viewpoints field
          alternativeViewpoint = completeData.alternative_viewpoints || undefined;

          // Store other metadata
          metadata = completeData.metadata;
          requestId = completeData.request_id;
          isComplete = true;

          // Call the complete callback if provided
          if (onCompleteReceived) {
            const result: StreamProcessResult = {
              content: fullContent,
              isComplete: true,
              sources,
              alternativeViewpoint,
              metadata,
              request_id: requestId
            };

            console.log('Calling onCompleteReceived with result:', {
              contentLength: result.content.length,
              hasSources: result.sources && result.sources.length > 0,
              sourcesCount: result.sources?.length || 0,
              hasAlternativeViewpoint: !!result.alternativeViewpoint,
              alternativeViewpointLength: typeof result.alternativeViewpoint === 'string' ? (result.alternativeViewpoint as string).length : 0,
              hasMetadata: !!result.metadata
            });

            onCompleteReceived(result);
          }

          // Also update the UI with the complete content
          onChunkReceived?.(fullContent);
        }
      } catch (error) {
        console.error('Error parsing event data:', error, data);
      }
    };

    await processEvents();

    // Log for debugging
    console.log('Stream processing completed:', {
      contentLength: fullContent.length,
      hasMetadata: !!metadata,
      hasSources: sources.length > 0,
      sourcesCount: sources.length,
      hasAlternativeViewpoint: !!alternativeViewpoint,
      alternativeViewpointLength: typeof alternativeViewpoint === 'string' ? (alternativeViewpoint as string).length : 0,
      isComplete
    });

    return {
      content: fullContent,
      isComplete,
      sources,
      alternativeViewpoint,
      metadata,
      request_id: requestId
    };
  } catch (error) {
    console.error('Error processing stream with metadata:', error);
    throw error;
  }
}
