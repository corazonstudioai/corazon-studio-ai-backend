import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders, extractVideoUrl } from "../_shared/video.ts";

function json(origin: string | null, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(origin), "Content-Type": "application/json" },
  });
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
  const { data: userData } = await userClient.auth.getUser();
  if (!userData.user) {
    return json(origin, { message: "Tu sesión ya no es válida." }, 401);
  }

  const body = await request.json().catch(() => ({}));
  const jobId = typeof body.job_id === "string" ? body.job_id : "";
  if (!jobId) {
    return json(origin, { message: "Falta el identificador del video." }, 400);
  }

  const service = createClient(url, serviceKey);
  const { data: job } = await service.from("video_jobs").select("*")
    .eq("id", jobId).eq("user_id", userData.user.id).maybeSingle();
  if (!job) return json(origin, { message: "No encontramos ese video." }, 404);
  if (job.status === "completed") {
    return json(origin, { status: "completed", video_url: job.video_url });
  }
  if (job.status === "failed") {
    return json(origin, {
      status: "failed",
      message: "No pudimos completar el video.",
    });
  }

  let providerStatus: Response;
  try {
    providerStatus = await fetch(job.status_url, {
      headers: { "Authorization": `Key ${falKey}` },
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    return json(origin, {
      status: job.status,
      message: "El video sigue procesándose.",
    }, 202);
  }
  if (!providerStatus.ok) {
    return json(origin, {
      status: job.status,
      message: "El video sigue procesándose.",
    }, 202);
  }
  const state = await providerStatus.json();
  const falStatus = String(state.status ?? "").toUpperCase();

  if (falStatus === "IN_QUEUE" || falStatus === "IN_PROGRESS") {
    const next = falStatus === "IN_QUEUE" ? "queued" : "in_progress";
    await service.from("video_jobs").update({
      status: next,
      updated_at: new Date().toISOString(),
    }).eq("id", job.id);
    return json(origin, { status: next }, 202);
  }

  if (falStatus === "COMPLETED") {
    let result: unknown = null;
    try {
      const resultResponse = await fetch(job.response_url, {
        headers: { "Authorization": `Key ${falKey}` },
        signal: AbortSignal.timeout(15_000),
      });
      result = resultResponse.ok ? await resultResponse.json() : null;
    } catch {
      result = null;
    }
    const videoUrl = extractVideoUrl(result);
    if (videoUrl) {
      await service.from("video_jobs").update({
        status: "completed",
        video_url: videoUrl,
        updated_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
      }).eq("id", job.id);
      await service.rpc("finalize_generation", {
        p_reservation_id: job.reservation_id,
        p_success: true,
      });
      return json(origin, { status: "completed", video_url: videoUrl });
    }
  }

  if (["FAILED", "CANCELLED"].includes(falStatus)) {
    await service.from("video_jobs").update({
      status: "failed",
      error_code: falStatus.toLowerCase(),
      updated_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
    }).eq("id", job.id);
    await service.rpc("finalize_generation", {
      p_reservation_id: job.reservation_id,
      p_success: false,
    });
    return json(origin, {
      status: "failed",
      message: "No pudimos completar el video. Tus créditos fueron devueltos.",
    });
  }

  return json(origin, {
    status: job.status,
    message: "El video sigue procesándose.",
  }, 202);
});
