import { corsHeaders } from "../_shared/video.ts";
import { parseSpeechRequest } from "../_shared/content.ts";
import {
  authenticateAndReserve,
  bytesToBase64,
  failureResponse,
  finalize,
  isUnsafe,
  json,
  openAi,
  requestId,
} from "../_shared/ai.ts";

Deno.serve(async (request) => {
  const origin = request.headers.get("origin");
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders(origin) });
  }
  if (request.method !== "POST") {
    return json(origin, { message: "Método no permitido." }, 405);
  }

  let service;
  let reservationId = "";
  try {
    const body = await request.json();
    const { text, voice, instructions } = parseSpeechRequest(body);
    if (isUnsafe(text)) {
      return json(origin, {
        message: "Esta solicitud no cumple las reglas de contenido seguro.",
      }, 400);
    }
    const model = Deno.env.get("OPENAI_TTS_MODEL") ?? "gpt-4o-mini-tts";
    const minutes = Math.max(text.length / 900, 0.01);
    const reserved = await authenticateAndReserve(request, {
      resource: "voice_minutes",
      units: minutes,
      creditCost: 1,
      estimatedCost: Number(Deno.env.get("TTS_ESTIMATED_COST_USD") ?? "0.02") *
        minutes,
      engine: model,
      idempotencyKey: requestId(body.idempotency_key),
    });
    service = reserved.service;
    reservationId = reserved.reservation.reservation_id;

    const response = await openAi("audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        voice,
        input: text,
        response_format: "mp3",
        instructions: instructions || undefined,
      }),
    });
    if (!response.ok) throw new Error("provider_error");
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (!bytes.length) throw new Error("provider_error");
    const audio = bytesToBase64(bytes);
    await finalize(service, reservationId, true);
    return json(origin, {
      status: "ok",
      audio_url: `data:audio/mpeg;base64,${audio}`,
      credits_remaining: reserved.reservation.credits_remaining,
    });
  } catch (error) {
    if (service && reservationId) await finalize(service, reservationId, false);
    return failureResponse(origin, error);
  }
});
