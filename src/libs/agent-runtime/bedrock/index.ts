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
import { experimental_buildLlama2Prompt } from 'ai/prompts';

import type { ChatModelCard } from '@/types/llm';

import { LobeRuntimeAI } from '../BaseAI';
import { AgentRuntimeErrorType } from '../error';
import {
  ChatCompetitionOptions,
  ChatStreamPayload,
  Embeddings,
  EmbeddingsOptions,
  EmbeddingsPayload,
  ModelProvider,
} from '../types';
import { buildAnthropicMessages, buildAnthropicTools } from '../utils/anthropicHelpers';
import { AgentRuntimeError } from '../utils/createError';
import { debugStream } from '../utils/debugStream';
import { StreamingResponse } from '../utils/response';
import {
  AWSBedrockClaudeStream,
  AWSBedrockLlamaStream,
  createBedrockConverseStream,
  createBedrockStream,
} from '../utils/streams';

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

  async chat(payload: ChatStreamPayload, options?: ChatCompetitionOptions) {
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
    options?: ChatCompetitionOptions,
  ): Promise<Response> => {
    const { max_tokens, messages, model, temperature, top_p, tools } = payload;
    const system_message = messages.find((m) => m.role === 'system');
    const user_messages = messages.filter((m) => m.role !== 'system');

    const command = new InvokeModelWithResponseStreamCommand({
      accept: 'application/json',
      body: JSON.stringify({
        anthropic_version: 'bedrock-2023-05-31',
        max_tokens: max_tokens || 4096,
        messages: await buildAnthropicMessages(user_messages),
        system: system_message?.content as string,
        temperature: temperature / 2,
        tools: buildAnthropicTools(tools),
        top_p: top_p,
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
    options?: ChatCompetitionOptions,
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

  private invokeNovaModel = async (
    payload: ChatStreamPayload,
    options?: ChatCompetitionOptions,
  ): Promise<Response> => {
    const modelId = payload.model;
    const newMessages = [];
    const systemMessages = [];
    let isFirstMessage = true;

    for (const message of payload.messages) {
      if (message.role === 'system') {
        systemMessages.push({ text: message.content.toString() });
      } else {
        // make sure the first message is a user message.
        if (isFirstMessage && message.role !== 'user') {
          isFirstMessage = false;
          continue;
        }
        isFirstMessage = false;

        const crole = message.role === 'user' ? ConversationRole.USER : ConversationRole.ASSISTANT;
        const newMess = {
          content: [{ text: message.content.toString() }],
          role: crole,
        };
        newMessages.push(newMess);
      }
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
      system: systemMessages,
    };

    // Step 5: Send and process the streaming request
    // - Send the request to the model
    // - Process each chunk of the streaming response
    try {
      // console.log('request', request);
      const response = await this.client.send(new ConverseStreamCommand(request));
      const claudeStream = createBedrockConverseStream(response);

      const [prod, debug] = claudeStream.tee();

      if (process.env.DEBUG_BEDROCK_CHAT_COMPLETION === '1') {
        debugStream(debug).catch(console.error);
      }

      // Respond with the stream
      return StreamingResponse(AWSBedrockClaudeStream(prod, options?.callback), {
        headers: options?.headers,
      });
    } catch (e) {
      console.log('Got an error. This is the request:', request);
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
