import Anthropic from '@anthropic-ai/sdk';
import { InvokeModelWithResponseStreamResponse } from '@aws-sdk/client-bedrock-runtime';

import { ChatStreamCallbacks } from '../../../types';
import { nanoid } from '../../../utils/uuid';
// import { transformAnthropicStream } from '../anthropic';
import {
  StreamContext,
  StreamProtocolChunk,
  createCallbacksTransformer,
  createSSEProtocolTransformer,
} from '../protocol';
import { createBedrockStream } from './common';

const transformAnthropicStream2 = (
  chunk: Anthropic.MessageStreamEvent,
  stack: StreamContext,
): StreamProtocolChunk | StreamProtocolChunk[] => {
  return { data: chunk, id: stack.id, type: 'text' };
};

export const AWSBedrockClaudeStream = (
  res: InvokeModelWithResponseStreamResponse | ReadableStream,
  cb?: ChatStreamCallbacks,
): ReadableStream<string> => {
  const streamStack: StreamContext = { id: 'chat_' + nanoid() };

  const stream = res instanceof ReadableStream ? res : createBedrockStream(res);

  return stream
    .pipeThrough(createSSEProtocolTransformer(transformAnthropicStream2, streamStack))
    .pipeThrough(createCallbacksTransformer(cb));
};
