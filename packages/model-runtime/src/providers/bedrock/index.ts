import {
  BedrockClient,
  FoundationModelSummary,
  ListFoundationModelsCommand,
} from '@aws-sdk/client-bedrock';
import {
  BedrockRuntimeClient,
  ConversationRole,
  ConverseStreamCommand,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import { ModelProvider } from '@lobehub/icons';

import type { ChatModelCard } from '@/types/llm';

import { LobeRuntimeAI } from '../../core/BaseAI';
import { buildAnthropicMessages, buildAnthropicTools } from '../../core/contextBuilders/anthropic';
import { MODEL_PARAMETER_CONFLICTS, resolveParameters } from '../../core/parameterResolver';
import {
  AWSBedrockClaudeStream,
  AWSBedrockLlamaStream,
  createBedrockConverseStream,
  createBedrockStream,
} from '../../core/streams/bedrock';
import {
  ChatMethodOptions,
  ChatStreamPayload,
  Embeddings,
  EmbeddingsOptions,
  EmbeddingsPayload,
  UserMessageContentPart,
} from '../../types';
import { AgentRuntimeErrorType } from '../../types/error';
import { AgentRuntimeError } from '../../utils/createError';
import { debugStream } from '../../utils/debugStream';
import { StreamingResponse } from '../../utils/response';

/**
 * A prompt constructor for HuggingFace LLama 2 chat models.
 * Does not support `function` messages.
 * @see https://huggingface.co/meta-llama/Llama-2-70b-chat-hf and https://huggingface.co/blog/llama2#how-to-prompt-llama-2
 */
export function experimental_buildLlama2Prompt(messages: { content: string; role: string }[]) {
  const startPrompt = `<s>[INST] `;
  const endPrompt = ` [/INST]`;
  const conversation = messages.map(({ content, role }, index) => {
    switch (role) {
      case 'user': {
        return content.trim();
      }
      case 'assistant': {
        return ` [/INST] ${content}</s><s>[INST] `;
      }
      case 'function': {
        throw new Error('Llama 2 does not support function calls.');
      }
      default: {
        if (role === 'system' && index === 0) {
          return `<<SYS>>\n${content}\n<</SYS>>\n\n`;
        } else {
          throw new Error(`Invalid message role: ${role}`);
        }
      }
    }
  });

  return startPrompt + conversation.join('') + endPrompt;
}

export interface BedrockModelCard {
  displayName: string;
  inputTokenLimit: number;
  name: string;
  outputTokenLimit: number;
}

export interface LobeBedrockAIParams {
  accessKeyId?: string;
  accessKeySecret?: string;
  region?: string;
  sessionToken?: string;
}

export class LobeBedrockAI implements LobeRuntimeAI {
  private client: BedrockRuntimeClient;
  private bedrockClient: BedrockClient;

  region: string;

  constructor({ region, accessKeyId, accessKeySecret, sessionToken }: LobeBedrockAIParams = {}) {
    if (!(accessKeyId && accessKeySecret))
      throw AgentRuntimeError.createError(AgentRuntimeErrorType.InvalidBedrockCredentials);
    this.region = region ?? 'us-east-1';
    this.client = new BedrockRuntimeClient({
      credentials: {
        accessKeyId: accessKeyId,
        secretAccessKey: accessKeySecret,
        sessionToken: sessionToken,
      },
      region: this.region,
    });
    this.bedrockClient = new BedrockClient({
      credentials: {
        accessKeyId: accessKeyId,
        secretAccessKey: accessKeySecret,
        sessionToken: sessionToken,
      },
      region: this.region,
    });
  }

  async models() {
    console.log('call bedrock models');

    const input = {
      // byProvider: 'STRING_VALUE',
      // byCustomizationType: 'FINE_TUNING' || 'CONTINUED_PRE_TRAINING',
      // byOutputModality: 'TEXT' || 'IMAGE' || 'EMBEDDING',
      // byInferenceType: 'ON_DEMAND' || 'PROVISIONED',
    };

    const command = new ListFoundationModelsCommand(input);

    const response = await this.bedrockClient.send(command);

    const modelList = response.modelSummaries || [];

    return modelList
      .map((model: FoundationModelSummary) => {
        let modelName = model.modelId?.replace(/^models\//, '');
        if (modelName?.includes('amazon.nova')) {
          modelName = modelName.replace('amazon.nova', 'us.amazon.nova');
        }
        return {
          contextWindowTokens: 100_000,
          displayName: model.modelName,
          enabled: false,
          functionCall: modelName?.toLowerCase().includes('gemini'),
          id: modelName,
          vision:
            modelName?.toLowerCase().includes('nova-pro') ||
            modelName?.toLowerCase().includes('nova-lite'),
        };
      })
      .filter(Boolean) as ChatModelCard[];
  }

  async chat(payload: ChatStreamPayload, options?: ChatMethodOptions) {
    console.log('payload', payload);
    if (payload.model.startsWith('meta')) return this.invokeLlamaModel(payload, options);
    if (payload.model.includes('nova')) return this.invokeNovaModel(payload, options);

    return this.invokeClaudeModel(payload, options);
  }
  /**
   * Supports the Amazon Titan Text models series.
   * Cohere Embed models are not supported
   * because the current text size per request
   * exceeds the maximum 2048 characters limit
   * for a single request for this series of models.
   * [bedrock embed guide] https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
   */
  async embeddings(payload: EmbeddingsPayload, options?: EmbeddingsOptions): Promise<Embeddings[]> {
    const input = Array.isArray(payload.input) ? payload.input : [payload.input];
    const promises = input.map((inputText: string) =>
      this.invokeEmbeddingModel(
        {
          dimensions: payload.dimensions,
          input: inputText,
          model: payload.model,
        },
        options,
      ),
    );
    return Promise.all(promises);
  }

  private invokeEmbeddingModel = async (
    payload: EmbeddingsPayload,
    options?: EmbeddingsOptions,
  ): Promise<Embeddings> => {
    const command = new InvokeModelCommand({
      accept: 'application/json',
      body: JSON.stringify({
        dimensions: payload.dimensions,
        inputText: payload.input,
        normalize: true,
      }),
      contentType: 'application/json',
      modelId: payload.model,
    });
    try {
      const res = await this.client.send(command, { abortSignal: options?.signal });
      const responseBody = JSON.parse(new TextDecoder().decode(res.body));
      return responseBody.embedding;
    } catch (e) {
      const err = e as Error & { $metadata: any };
      throw AgentRuntimeError.chat({
        error: {
          body: err.$metadata,
          message: err.message,
          type: err.name,
        },
        errorType: AgentRuntimeErrorType.ProviderBizError,
        provider: ModelProvider.Bedrock,
        region: this.region,
      });
    }
  };

  private invokeClaudeModel = async (
    payload: ChatStreamPayload,
    options?: ChatMethodOptions,
  ): Promise<Response> => {
    const { max_tokens, messages, model, temperature, top_p, tools } = payload;
    const system_message = messages.find((m) => m.role === 'system');
    const user_messages = messages.filter((m) => m.role !== 'system');

    // Resolve temperature and top_p parameters based on model constraints
    const hasConflict = MODEL_PARAMETER_CONFLICTS.BEDROCK_CLAUDE_4_PLUS.has(model);
    const resolvedParams = resolveParameters(
      { temperature, top_p },
      { hasConflict, normalizeTemperature: true, preferTemperature: true },
    );

    const command = new InvokeModelWithResponseStreamCommand({
      accept: 'application/json',
      body: JSON.stringify({
        anthropic_version: 'bedrock-2023-05-31',
        max_tokens: max_tokens || 4096,
        messages: await buildAnthropicMessages(user_messages),
        system: system_message?.content as string,
        temperature: resolvedParams.temperature,
        tools: buildAnthropicTools(tools),
        top_p: resolvedParams.top_p,
      }),
      contentType: 'application/json',
      modelId: model,
    });

    try {
      // Ask Claude for a streaming chat completion given the prompt
      const res = await this.client.send(command, { abortSignal: options?.signal });

      const claudeStream = createBedrockStream(res);

      const [prod, debug] = claudeStream.tee();

      if (process.env.DEBUG_BEDROCK_CHAT_COMPLETION === '1') {
        debugStream(debug).catch(console.error);
      }

      // Respond with the stream
      return StreamingResponse(AWSBedrockClaudeStream(prod, options?.callback), {
        headers: options?.headers,
      });
    } catch (e) {
      const err = e as Error & { $metadata: any };

      throw AgentRuntimeError.chat({
        error: {
          body: err.$metadata,
          message: err.message,
          type: err.name,
        },
        errorType: AgentRuntimeErrorType.ProviderBizError,
        provider: ModelProvider.Bedrock,
        region: this.region,
      });
    }
  };

  private invokeLlamaModel = async (
    payload: ChatStreamPayload,
    options?: ChatMethodOptions,
  ): Promise<Response> => {
    const { max_tokens, messages, model } = payload;
    const command = new InvokeModelWithResponseStreamCommand({
      accept: 'application/json',
      body: JSON.stringify({
        max_gen_len: max_tokens || 400,
        prompt: experimental_buildLlama2Prompt(messages as any),
      }),
      contentType: 'application/json',
      modelId: model,
    });

    try {
      // Ask Claude for a streaming chat completion given the prompt
      const res = await this.client.send(command);

      const stream = createBedrockStream(res);

      const [prod, debug] = stream.tee();

      if (process.env.DEBUG_BEDROCK_CHAT_COMPLETION === '1') {
        debugStream(debug).catch(console.error);
      }
      // Respond with the stream
      return StreamingResponse(AWSBedrockLlamaStream(prod, options?.callback), {
        headers: options?.headers,
      });
    } catch (e) {
      const err = e as Error & { $metadata: any };

      throw AgentRuntimeError.chat({
        error: {
          body: err.$metadata,
          message: err.message,
          region: this.region,
          type: err.name,
        },
        errorType: AgentRuntimeErrorType.ProviderBizError,
        provider: ModelProvider.Bedrock,
        region: this.region,
      });
    }
  };

  /**
   * Extract text content from message content (string or array of content parts)
   */
  private extractTextContent = (content: string | UserMessageContentPart[]): string => {
    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      // Extract text from all text parts and join them
      const textParts = content
        .filter((part) => part.type === 'text' && part.text)
        .map((part) => (part as { text: string; type: 'text' }).text);

      return textParts.join('\n');
    }

    return '';
  };

  private invokeNovaModel = async (
    payload: ChatStreamPayload,
    options?: ChatMethodOptions,
  ): Promise<Response> => {
    const modelId = payload.model;
    const newMessages = [];

    // Filter out system messages and process the rest
    for (const message of payload.messages) {
      // Skip system messages as Nova handles them differently if needed
      if (message.role === 'system') {
        continue;
      }

      const textContent = this.extractTextContent(message.content);

      // Skip messages with empty content
      if (!textContent.trim()) {
        continue;
      }

      const crole = message.role === 'user' ? ConversationRole.USER : ConversationRole.ASSISTANT;
      const newMess = {
        content: [{ text: textContent }],
        role: crole,
      };
      newMessages.push(newMess);
    }
    // Step 4: Configure the streaming request
    // Optional parameters to control the model's response:
    // - maxTokens: maximum number of tokens to generate
    // - temperature: randomness (max: 1.0, default: 0.7)
    //   OR
    // - topP: diversity of word choice (max: 1.0, default: 0.9)
    // Note: Use either temperature OR topP, but not both
    const request = {
      inferenceConfig: {
        maxTokens: 4096, // The maximum response length
        temperature: payload.temperature, // Using temperature for randomness control
        //topP: 0.9,        // Alternative: use topP instead of temperature
      },
      messages: newMessages,
      modelId,
    };

    // Step 5: Send and process the streaming request
    // - Send the request to the model
    // - Process each chunk of the streaming response
    try {
      const response = await this.client.send(new ConverseStreamCommand(request));
      const claudeStream = createBedrockConverseStream(response);

      const prod = claudeStream.tee()[0];

      // if (process.env.DEBUG_BEDROCK_CHAT_COMPLETION === '1') {
      //   debugStream(_debug).catch(console.error);
      // }

      // Respond with the stream
      return StreamingResponse(AWSBedrockClaudeStream(prod, options?.callback), {
        headers: options?.headers,
      });
    } catch (e) {
      const err = e as Error & { $metadata: any };

      throw AgentRuntimeError.chat({
        error: {
          body: err.$metadata,
          message: err.message,
          region: this.region,
          type: err.name,
        },
        errorType: AgentRuntimeErrorType.ProviderBizError,
        provider: ModelProvider.Bedrock,
        region: this.region,
      });
    }
  };
}

export default LobeBedrockAI;
