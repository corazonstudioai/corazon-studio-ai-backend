import { readJsonRequest } from "../_shared/request.ts";
import { hashRequest, isUnsafe } from "../_shared/ai.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders, parseVideoRequest } from "../_shared/video.ts";
import {
  falSingleAttemptHeaders,
  lockedVideoModelError,
  OPENAI_VOICE_MODEL,
  WAN_TURBO_MODEL,
} from "../_shared/video_test_policy.ts";

function json(origin: string | null, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(origin), "Content-Type": "application/json" },
  });
}

async function refundStart(service: any, reservationId: string, origin: string | null, message: string, status: number, code?: string) {
  const { error } = await service.rpc("finalize_generation", { p_reservation_id: reservationId, p_success: false });
  if (error) return json(origin, { message: "No pudimos confirmar el estado de tus créditos." }, 503);
  return json(origin, { message, code }, status);
}

Deno.serve(async (request) => {
  const origin = request.headers.get("origin");
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders(origin) });
  }
  if (request.method !== "POST") {
    return json(origin, { message: "Método no permitido." }, 405);
  }

  const url = Deno.env.get("SUPABASE_URL");
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const falKey = Deno.env.get("FAL_KEY");
  if (!url || !anonKey || !serviceKey || !falKey) {
    return json(origin, {
      message: "El servicio de video no está disponible en este momento.",
    }, 503);
  }

  const auth = request.headers.get("authorization") ?? "";
  const userClient = createClient(url, anonKey, {
    global: { headers: { Authorization: auth } },
  });
  const { data: userData, error: userError } = await userClient.auth.getUser();
  const user = userData.user;
  if (userError || !user) {
    return json(origin, {
      message: "Verifica tu correo o teléfono para continuar.",
    }, 401);
  }
  const identityKind = user.email_confirmed_at
    ? "email"
    : user.phone_confirmed_at
    ? "phone"
    : null;
  if (!identityKind) {
    return json(origin, {
      message: "Verifica tu correo o teléfono para continuar.",
    }, 403);
  }

  const lockedTest = user.app_metadata?.locked_video_test === true;
  const model = Deno.env.get("VIDEO_MODEL_CINE") ?? WAN_TURBO_MODEL;
  const voiceModel = Deno.env.get("OPENAI_TTS_MODEL") ?? OPENAI_VOICE_MODEL;
  let input;
  let body;
  try {
    body = await readJsonRequest(request);
    if (lockedTest) {
      const code = lockedVideoModelError(model, voiceModel);
      if (code) {
        return json(origin, { code, message: "La prueba se detuvo: el motor configurado no coincide con el autorizado." }, 409);
      }
      if (!Deno.env.get("OPENAI_API_KEY")) {
        return json(origin, { code: "tts_unavailable", message: "El servicio de voz no está disponible." }, 503);
      }
      if (body.preflight === true) return json(origin, { locked_video_test: true });
    }
    input = parseVideoRequest(body);
    if (isUnsafe(input.prompt) || isUnsafe(input.narration)) throw new Error("unsafe_request");
  } catch {
    return json(origin, {
      message: "Revisa la descripción y la duración del video.",
    }, 400);
  }
  if (lockedTest && (input.duration !== 5 || input.audioMode !== "narrated")) {
    return json(origin, { code: "invalid_locked_video_test", message: "La prueba requiere cinco segundos y narración." }, 400);
  }

  if (input.audioMode === "narrated" && !Deno.env.get("OPENAI_API_KEY")) return json(origin, {message:"El servicio de voz no está disponible en este momento."}, 503);
  const service = createClient(url, serviceKey);
  const perSecond = Number(
    Deno.env.get("FAL_ESTIMATED_COST_PER_SECOND") ?? "0.05",
  );
  const estimatedCost = Math.max(0, perSecond * input.duration);

  const { data: reservations, error: reserveError } = await service.rpc(
    "reserve_generation_v2",
    {
      p_user_id: user.id,
      p_identity_kind: identityKind,
      p_resource: "video_minutes",
      p_units: input.duration / 60,
      p_credit_cost: 1,
      p_estimated_cost_usd: estimatedCost,
      p_engine: model,
      p_idempotency_key: input.idempotencyKey,
      p_request_hash: await hashRequest(body),
    },
  );
  const reservation = reservations?.[0];
  if (reserveError || !reservation?.allowed) {
    const reason = reservation?.reason;
    const message = reason === "credits_exhausted"
      ? "Tus créditos se terminaron. Compra créditos para crear otro video."
      : reason === "resource_limit_reached"
      ? "Llegaste al límite de video de tu plan."
      : "No pudimos verificar tus créditos en este momento.";
    return json(
      origin,
      { message, reason },
      reason === "credits_exhausted" ? 402 : 403,
    );
  }

  const previous = await service.from("video_jobs").select("id,status").eq(
    "reservation_id",
    reservation.reservation_id,
  ).maybeSingle();
  if (previous.error) {
    if (reservation.reason === "reserved") return refundStart(service, reservation.reservation_id, origin, "No pudimos iniciar el seguimiento. Tu crédito fue devuelto.", 503);
    return json(origin, { code: "tracking_unavailable", message: "No pudimos consultar el seguimiento de esta solicitud." }, 503);
  }
  if (previous.data) {
    return json(origin, {
      status: previous.data.status,
      job_id: previous.data.id,
      credits_remaining: reservation.credits_remaining,
    }, 202);
  }

  if (reservation.reason === "idempotent_replay") {
    return json(origin, { code: "generation_in_progress", message: "Tu solicitud sigue procesándose. No necesitas crearla otra vez." }, 409);
  }

  let falResponse: Response;
  try {
    const providerInput = lockedTest
      ? { prompt: input.prompt, aspect_ratio: input.aspectRatio, resolution: "480p", enable_prompt_expansion: false }
      : { prompt: input.prompt, duration: input.duration, aspect_ratio: input.aspectRatio };
    falResponse = await fetch(`https://queue.fal.run/${lockedTest ? WAN_TURBO_MODEL : model}`, {
      method: "POST",
      headers: {
        "Authorization": `Key ${falKey}`,
        "Content-Type": "application/json",
        ...(lockedTest ? falSingleAttemptHeaders : {}),
      },
      body: JSON.stringify(providerInput),
      signal: AbortSignal.timeout(30_000),
    });
  } catch {
    return refundStart(service, reservation.reservation_id, origin, "El motor de video tardó demasiado. Tus créditos fueron devueltos.", 504, lockedTest ? "fal_video_request_timeout_or_connection" : undefined);
  }
  if (!falResponse.ok) {
    return refundStart(service, reservation.reservation_id, origin, "El motor de video no respondió. Tus créditos fueron devueltos.", 502, lockedTest ? `fal_video_submit_http_${falResponse.status}` : undefined);
  }

  const queued = await falResponse.json().catch(() => ({}));
  if (!queued.request_id || !queued.status_url || !queued.response_url) {
    return refundStart(service, reservation.reservation_id, origin, "No pudimos iniciar el video. Tus créditos fueron devueltos.", 502);
  }

  const { data: job, error: jobError } = await service.from("video_jobs")
    .insert({
      user_id: user.id,
      reservation_id: reservation.reservation_id,
      model,
      provider_request_id: queued.request_id,
      status_url: queued.status_url,
      response_url: queued.response_url,
      duration_seconds: input.duration,
      narration_text: input.narration,
      audio_mode: input.audioMode,
      voice: input.voice,
    }).select("id,status").single();

  if (jobError || !job) {
    return refundStart(service, reservation.reservation_id, origin, "El video inició, pero no pudimos guardar su seguimiento. Tus créditos fueron devueltos.", 500);
  }
  return json(origin, {
    status: job.status,
    job_id: job.id,
    credits_remaining: reservation.credits_remaining,
  }, 202);
});
