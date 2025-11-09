import { EdgeSpeechPayload, EdgeSpeechTTS } from '@lobehub/tts';

export const runtime = 'nodejs';

export const POST = async (req: Request) => {
  const payload = (await req.json()) as EdgeSpeechPayload;

  return await EdgeSpeechTTS.createRequest({ payload });
};
