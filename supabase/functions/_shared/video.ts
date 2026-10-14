export const allowedOrigins = new Set([
  "https://corazonstudioai.github.io",
  "http://localhost:5173",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
]);

export function corsHeaders(origin: string | null) {
  const selected = origin && allowedOrigins.has(origin)
    ? origin
    : "https://corazonstudioai.github.io";
  return {
    "Access-Control-Allow-Origin": selected,
    "Access-Control-Allow-Headers":
      "authorization, x-client-info, apikey, content-type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Vary": "Origin",
  };
}

export function parseVideoRequest(value: unknown) {
  if (!value || typeof value !== "object") throw new Error("invalid_request");
  const body = value as Record<string, unknown>;
  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
  const duration = Number(body.duration ?? 5);
  const idempotencyKey = typeof body.idempotency_key === "string"
    ? body.idempotency_key
    : crypto.randomUUID();
  const aspectRatio = body.aspect_ratio === undefined ? "16:9" : body.aspect_ratio;
  if ((typeof aspectRatio !== "string" || !["9:16","16:9","1:1"].includes(aspectRatio))) throw new Error("invalid_aspect_ratio");
  const audioMode = body.audio_mode === undefined ? "narrated" : body.audio_mode;
  if ((typeof audioMode !== "string" || !["narrated","silent"].includes(audioMode))) throw new Error("invalid_audio_mode");
  const narration = typeof body.narration === "string"
    ? body.narration.trim()
    : audioMode === "silent" ? "" : prompt;
  const voice = body.voice === "male" || body.voice === "onyx"
    ? "onyx"
    : "nova";
  if (prompt.length < 3 || prompt.length > 1500) {
    throw new Error("invalid_prompt");
  }
  if (![5, 10].includes(duration)) throw new Error("invalid_duration");
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
      .test(idempotencyKey)
  ) {
    throw new Error("invalid_idempotency_key");
  }
  if ((audioMode === "narrated" && !narration) || (audioMode === "silent" && !!narration) || narration.length > 1200 || (audioMode === "narrated" && narration.split(/\s+/).length > (duration <= 5 ? 12 : 25))) {
    throw new Error("invalid_narration");
  }
  return { prompt, duration, idempotencyKey, narration, voice, audioMode, aspectRatio };
}

export function narrationForDuration(text: string, duration: number) {
  const maxWords = duration <= 5 ? 12 : 25;
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length || words.length > maxWords) throw new Error("invalid_narration");
  return text.trim();
}

export function extractVideoUrl(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const data = payload as Record<string, unknown>;
  if (typeof data.video_url === "string") return data.video_url;
  const video = data.video as Record<string, unknown> | undefined;
  if (typeof video?.url === "string") return video.url;
  const videos = data.videos;
  if (Array.isArray(videos)) {
    const first = videos[0] as Record<string, unknown> | undefined;
    if (typeof first?.url === "string") return first.url;
  }
  return null;
}

// A separate visual timeline keeps short speech from truncating the video.
export function videoAudioTimeline(videoUrl: string, audioUrl: string, durationSeconds: number, speechDurationSeconds: number) {
  if (![5, 10].includes(durationSeconds)) throw new Error("invalid_duration");
  if (!Number.isFinite(speechDurationSeconds) || speechDurationSeconds <= 0 || speechDurationSeconds > durationSeconds) throw new Error("narration_does_not_fit");
  const duration = durationSeconds * 1000;
  return { tracks: [
    { id: "visual", type: "video", keyframes: [{ url: videoUrl, timestamp: 0, duration }] },
    { id: "narration", type: "audio", keyframes: [{ url: audioUrl, timestamp: 0, duration: speechDurationSeconds * 1000 }] },
  ] };
}

// OpenAI PCM is mono 24 kHz, signed little-endian 16-bit samples.
export function pcmSpeechAudio(pcm: Uint8Array) {
  if (!pcm.byteLength || pcm.byteLength % 2) throw new Error("invalid_speech_audio");
  const wav = new Uint8Array(44 + pcm.byteLength);
  const view = new DataView(wav.buffer);
  const text = (offset: number, value: string) => { for (let i=0;i<value.length;i++) wav[offset+i]=value.charCodeAt(i); };
  text(0,"RIFF"); view.setUint32(4,36+pcm.byteLength,true); text(8,"WAVE");
  text(12,"fmt "); view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
  view.setUint32(24,24000,true); view.setUint32(28,48000,true); view.setUint16(32,2,true); view.setUint16(34,16,true);
  text(36,"data"); view.setUint32(40,pcm.byteLength,true); wav.set(pcm,44);
  return {wav,durationSeconds:pcm.byteLength/48000};
}

export function silentVideoTimeline(videoUrl: string, durationSeconds: number) {
  if (![5,10].includes(durationSeconds)) throw new Error("invalid_duration");
  return {tracks:[{id:"visual",type:"video",keyframes:[{url:videoUrl,timestamp:0,duration:durationSeconds*1000}]}]};
}
