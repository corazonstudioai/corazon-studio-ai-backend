import { replayResult, saveMediaResult } from "../_shared/results.ts";
import { corsHeaders } from "../_shared/video.ts";
import {
  authenticateAndReserve,
  cleanText,
  failureResponse,
  finalize,
  isUnsafe,
  json,
  openAi,
  requestId,
  hashRequest,
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
    const body = await request.json().catch(() => null);
    if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("invalid_request");
    if (typeof body.prompt === "string" && body.prompt.trim().length > 1800) throw new Error("invalid_request");
    const prompt = cleanText(body.prompt, 1800);
    if (!prompt) {
      return json(
        origin,
        { message: "Describe la imagen que deseas crear." },
        400,
      );
    }
    if (isUnsafe(prompt)) {
      return json(origin, {
        message: "Esta solicitud no cumple las reglas de contenido seguro.",
      }, 400);
    }
    const model = Deno.env.get("OPENAI_IMAGE_MODEL") ?? "gpt-image-1";
    const size = ["1024x1024", "1536x1024", "1024x1536"].includes(body.size)
      ? body.size
      : "1024x1024";
    const reserved = await authenticateAndReserve(request, {
      resource: "images",
      units: 1,
      creditCost: 1,
      estimatedCost: Number(Deno.env.get("IMAGE_ESTIMATED_COST_USD") ?? "0.04"),
      engine: model,
      idempotencyKey: requestId(body.idempotency_key),
      requestHash: await hashRequest(body),
    });
    const replay = await replayResult(reserved.service, reserved.reservation);
    if (replay) return json(origin, { ...replay, credits_remaining: reserved.reservation.credits_remaining });
    service = reserved.service;
    reservationId = reserved.reservation.reservation_id;

    const response = await openAi("images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt:
          `${prompt}. High quality, coherent composition, no watermark, no extra text unless explicitly requested.`,
        size,
      }),
    });
    if (!response.ok) throw new Error("provider_error");
    const result = await response.json();
    const image = result.data?.[0]?.b64_json;
    if (typeof image !== "string" || !image) throw new Error("provider_error");
    const bytes = Uint8Array.from(atob(image), character => character.charCodeAt(0));
    const output = await saveMediaResult(service, reservationId, "image_url", bytes, "image/png");
    return json(origin, { ...output, credits_remaining: reserved.reservation.credits_remaining });
  } catch (error) {
    if (service && reservationId) {
      try { await finalize(service, reservationId, false); }
      catch { return failureResponse(origin, new Error("billing_unavailable")); }
    }
    return failureResponse(origin, error);
  }
});
