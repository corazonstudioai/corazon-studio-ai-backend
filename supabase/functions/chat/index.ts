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
    const message = cleanText(body.message, 4000);
    if (!message) {
      return json(origin, { message: "Escribe una idea para comenzar." }, 400);
    }
    if (isUnsafe(message)) {
      return json(origin, {
        message: "Esta solicitud no cumple las reglas de contenido seguro.",
      }, 400);
    }
    const model = Deno.env.get("OPENAI_MODEL") ?? "gpt-4o-mini";
    const reserved = await authenticateAndReserve(request, {
      resource: "text_requests",
      units: 1,
      creditCost: 1,
      estimatedCost: Number(Deno.env.get("TEXT_ESTIMATED_COST_USD") ?? "0.001"),
      engine: model,
      idempotencyKey: requestId(body.idempotency_key),
    });
    service = reserved.service;
    reservationId = reserved.reservation.reservation_id;

    const response = await openAi("chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content:
              "Eres Corazón Studio AI. Responde en el idioma del usuario con calidez, claridad y utilidad. Ayuda a crear contenido positivo, educativo, humano y seguro. No produzcas contenido sexual, explotador, violento ni instrucciones peligrosas.",
          },
          { role: "user", content: message },
        ],
        max_tokens: 700,
        temperature: 0.65,
      }),
    });
    if (!response.ok) throw new Error("provider_error");
    const result = await response.json();
    const reply = result.choices?.[0]?.message?.content ?? "";
    await finalize(service, reservationId, true);
    return json(origin, {
      status: "ok",
      reply,
      credits_remaining: reserved.reservation.credits_remaining,
    });
  } catch (error) {
    if (service && reservationId) await finalize(service, reservationId, false);
    return failureResponse(origin, error);
  }
});
