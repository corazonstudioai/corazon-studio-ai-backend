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
    });
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
    const imageUrl = image
      ? `data:image/png;base64,${image}`
      : result.data?.[0]?.url;
    if (!imageUrl) throw new Error("provider_error");
    await finalize(service, reservationId, true);
    return json(origin, {
      status: "ok",
      image_url: imageUrl,
      credits_remaining: reserved.reservation.credits_remaining,
    });
  } catch (error) {
    if (service && reservationId) await finalize(service, reservationId, false);
    return failureResponse(origin, error);
  }
});
