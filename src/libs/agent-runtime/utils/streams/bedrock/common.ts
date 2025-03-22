import {
  ConverseStreamOutput,
  ConverseStreamResponse,
  InvokeModelWithResponseStreamResponse,
  ResponseStream,
} from '@aws-sdk/client-bedrock-runtime';
import { readableFromAsyncIterable } from 'ai';

const chatStreamable = async function* (stream: AsyncIterable<ResponseStream>) {
  for await (const response of stream) {
    if (response.chunk) {
      const decoder = new TextDecoder();

      const value = decoder.decode(response.chunk.bytes, { stream: true });
      try {
        const chunk = JSON.parse(value);

        yield chunk;
      } catch (e) {
        console.log('bedrock stream parser error:', e);

        yield value;
      }
    } else {
      yield response;
    }
  }
};

const chatStreamable2 = async function* (stream: AsyncIterable<ConverseStreamOutput>) {
  for await (const response of stream) {
    if (response.contentBlockDelta) {
      try {
        const chunk = response.contentBlockDelta.delta?.text;
        console.log('chunk', chunk);
        yield chunk;
      } catch (e) {
        console.log('bedrock converse stream error:', e);

        yield '';
      }
    }
    // } else {
    //   yield response;
    // }
  }
};
/**
 * covert the bedrock response to a readable stream
 */
export const createBedrockStream = (res: InvokeModelWithResponseStreamResponse) =>
  readableFromAsyncIterable(chatStreamable(res.body!));

export const createBedrockConverseStream = (res: ConverseStreamResponse) =>
  readableFromAsyncIterable(chatStreamable2(res.stream!));
