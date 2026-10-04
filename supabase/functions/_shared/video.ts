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
  const narration = typeof body.narration === "string"
    ? body.narration.trim()
    : prompt;
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
  if (!narration || narration.length > 1200) {
    throw new Error("invalid_narration");
  }
  return { prompt, duration, idempotencyKey, narration, voice };
}

export function narrationForDuration(text: string, duration: number) {
  const maxWords = duration <= 5 ? 12 : 25;
  const words = text.trim().split(/\s+/).filter(Boolean);
  const shortened = words.slice(0, maxWords).join(" ");
  if (!shortened) throw new Error("invalid_narration");
  return words.length > maxWords ? `${shortened}…` : shortened;
}

export function extractVideoUrl(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const data = payload as Record<string, unknown>;
  const video = data.video as Record<string, unknown> | undefined;
  if (typeof video?.url === "string") return video.url;
  const videos = data.videos;
  if (Array.isArray(videos)) {
    const first = videos[0] as Record<string, unknown> | undefined;
    if (typeof first?.url === "string") return first.url;
  }
  return null;
}
