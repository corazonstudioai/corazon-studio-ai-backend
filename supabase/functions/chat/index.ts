import { replayResult, saveResult } from "../_shared/results.ts";
import { corsHeaders } from "../_shared/video.ts";
import { languageInstruction, requireReply } from "../_shared/content.ts";
import { brainInstructions, parseBrainReply } from "../_shared/brain.ts";
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
    if (typeof body.message === "string" && body.message.trim().length > 4000) throw new Error("invalid_request");
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
    const planning = body.purpose === "brain";
    const reserved = await authenticateAndReserve(request, {
      resource: "text_requests",
      units: 1,
      creditCost: 1,
      estimatedCost: Number(Deno.env.get("TEXT_ESTIMATED_COST_USD") ?? "0.001"),
      engine: model,
      idempotencyKey: requestId(body.idempotency_key),
      requestHash: await hashRequest(body),
    });
    const replay = await replayResult(reserved.service, reserved.reservation);
    if (replay) return json(origin, { ...replay, credits_remaining: reserved.reservation.credits_remaining });
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
            content: planning ? brainInstructions(body.language) : `Eres Corazón Studio AI. ${
              languageInstruction(body.language)
            } Responde con calidez, claridad y utilidad. Ayuda a crear contenido positivo, educativo, humano y seguro. No produzcas contenido sexual, explotador, violento ni instrucciones peligrosas.`,
          },
          { role: "user", content: planning ? JSON.stringify({ request: message, preferences: body.preferences ?? {} }) : message },
        ],
        max_tokens: planning ? 1100 : 700,
        temperature: planning ? 0.2 : 0.65,
        ...(planning ? { response_format: { type: "json_object" } } : {}),
      }),
    });
    if (!response.ok) throw new Error("provider_error");
    const result = await response.json();
    const reply = requireReply(result.choices?.[0]?.message?.content);
    const plan = planning ? parseBrainReply(reply) : undefined;
    const output = {
      status: "ok",
      reply: plan?.message ?? reply,
      ...(plan ? { plan } : {}),
    };
    await saveResult(service, reservationId, output);
    return json(origin, { ...output, credits_remaining: reserved.reservation.credits_remaining });
  } catch (error) {
    if (service && reservationId) {
      try { await finalize(service, reservationId, false); }
      catch { return failureResponse(origin, new Error("billing_unavailable")); }
    }
    return failureResponse(origin, error);
  }
});
