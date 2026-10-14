export function outputLanguage(value: unknown): "es" | "en" | null {
  return value === "es" || value === "en" ? value : null;
}

export function languageInstruction(value: unknown) {
  const language = outputLanguage(value);
  return language === "es"
    ? "Escribe el resultado en español."
    : language === "en"
    ? "Write the result in English."
    : "Responde en el idioma del usuario.";
}

export function parseSpeechRequest(value: unknown) {
  if (!value || typeof value !== "object") throw new Error("invalid_request");
  const body = value as Record<string, unknown>;
  const text = typeof body.text === "string" ? body.text.trim() : "";
  if (!text) throw new Error("empty_speech");
  if (text.length > 1200) throw new Error("speech_too_long");
  const voices = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
  ];
  const voice = typeof body.voice === "string" && voices.includes(body.voice)
    ? body.voice
    : "nova";
  const instructions = typeof body.instructions === "string"
    ? body.instructions.trim().slice(0, 500)
    : "";
  return { text, voice, instructions };
}

export function requireReply(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("provider_error");
  }
  return value.trim();
}
