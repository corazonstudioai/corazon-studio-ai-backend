import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders, parseVideoRequest } from "../_shared/video.ts";

function json(origin: string | null, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(origin), "Content-Type": "application/json" },
  });
}

Deno.serve(async (request) => {
  const origin = request.headers.get("origin");
  if (request.method === "OPTIONS") return new Response("ok", { headers: corsHeaders(origin) });
  if (request.method !== "POST") return json(origin, { message: "Método no permitido." }, 405);

  const url = Deno.env.get("SUPABASE_URL");
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const falKey = Deno.env.get("FAL_KEY");
  if (!url || !anonKey || !serviceKey || !falKey) {
    return json(origin, { message: "El servicio de video no está disponible en este momento." }, 503);
  }

  const auth = request.headers.get("authorization") ?? "";
  const userClient = createClient(url, anonKey, { global: { headers: { Authorization: auth } } });
  const { data: userData, error: userError } = await userClient.auth.getUser();
  const user = userData.user;
  if (userError || !user) return json(origin, { message: "Verifica tu correo o teléfono para continuar." }, 401);
  const identityKind = user.email_confirmed_at ? "email" : user.phone_confirmed_at ? "phone" : null;
  if (!identityKind) return json(origin, { message: "Verifica tu correo o teléfono para continuar." }, 403);

  let input;
  try {
    input = parseVideoRequest(await request.json());
  } catch {
    return json(origin, { message: "Revisa la descripción y la duración del video." }, 400);
  }

  const service = createClient(url, serviceKey);
  const model = Deno.env.get("VIDEO_MODEL_CINE") ?? "fal-ai/wan/v2.2-a14b/text-to-video/turbo";
  const perSecond = Number(Deno.env.get("FAL_ESTIMATED_COST_PER_SECOND") ?? "0.05");
  const estimatedCost = Math.max(0, perSecond * input.duration);

  const { data: reservations, error: reserveError } = await service.rpc("reserve_generation", {
    p_user_id: user.id,
    p_identity_kind: identityKind,
    p_resource: "video_minutes",
    p_units: input.duration / 60,
    p_credit_cost: 1,
    p_estimated_cost_usd: estimatedCost,
    p_engine: model,
    p_idempotency_key: input.idempotencyKey,
  });
  const reservation = reservations?.[0];
  if (reserveError || !reservation?.allowed) {
    const reason = reservation?.reason;
    const message = reason === "credits_exhausted"
      ? "Tus créditos se terminaron. Compra créditos para crear otro video."
      : reason === "resource_limit_reached"
      ? "Llegaste al límite de video de tu plan."
      : "No pudimos verificar tus créditos en este momento.";
    return json(origin, { message, reason }, reason === "credits_exhausted" ? 402 : 403);
  }

  const previous = await service.from("video_jobs").select("id,status").eq(
    "reservation_id",
    reservation.reservation_id,
  ).maybeSingle();
  if (previous.data) {
    return json(origin, {
      status: previous.data.status,
      job_id: previous.data.id,
      credits_remaining: reservation.credits_remaining,
    }, 202);
  }

  const falResponse = await fetch(`https://queue.fal.run/${model}`, {
    method: "POST",
    headers: { "Authorization": `Key ${falKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: input.prompt, duration: input.duration }),
  });
  if (!falResponse.ok) {
    await service.rpc("finalize_generation", {
      p_reservation_id: reservation.reservation_id,
      p_success: false,
    });
    return json(origin, { message: "El motor de video no respondió. Tus créditos fueron devueltos." }, 502);
  }

  const queued = await falResponse.json();
  if (!queued.request_id || !queued.status_url || !queued.response_url) {
    await service.rpc("finalize_generation", {
      p_reservation_id: reservation.reservation_id,
      p_success: false,
    });
    return json(origin, { message: "No pudimos iniciar el video. Tus créditos fueron devueltos." }, 502);
  }

  const { data: job, error: jobError } = await service.from("video_jobs").insert({
    user_id: user.id,
    reservation_id: reservation.reservation_id,
    model,
    provider_request_id: queued.request_id,
    status_url: queued.status_url,
    response_url: queued.response_url,
    duration_seconds: input.duration,
  }).select("id,status").single();

  if (jobError || !job) {
    return json(origin, { message: "El video inició, pero no pudimos guardar su seguimiento." }, 500);
  }
  return json(origin, {
    status: job.status,
    job_id: job.id,
    credits_remaining: reservation.credits_remaining,
  }, 202);
});
