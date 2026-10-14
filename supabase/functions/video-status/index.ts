import { readJsonRequest } from "../_shared/request.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import {
  corsHeaders,
  extractVideoUrl,
  narrationForDuration,
} from "../_shared/video.ts";

const MERGE_MODEL = "fal-ai/ffmpeg-api/merge-audio-video";

function json(origin: string | null, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders(origin), "Content-Type": "application/json" },
  });
}

async function settleVideo(service: any, job: Record<string, any>, origin: string | null, success: boolean, videoUrl: string | null = null, code: string | null = null) {
  const { data, error } = await service.rpc("settle_video_job", { p_job_id: job.id, p_success: success, p_video_url: videoUrl, p_error_code: code });
  if (error || !data) return json(origin, { message: "No pudimos confirmar el estado final y los créditos del video." }, 503);
  return json(origin, { ...data, ...(data.status === "failed" ? { message: "No pudimos completar el video. Tu crédito fue devuelto." } : {}) });
}

function queueUrl(value: unknown) {
  if (typeof value !== "string") throw new Error("invalid_provider_url");
  const url = new URL(value);
  if (url.protocol !== "https:" || url.hostname !== "queue.fal.run" || url.username || url.password) throw new Error("invalid_provider_url");
  return url.href;
}

async function queueAudioMerge(
  job: Record<string, any>,
  videoUrl: string,
  falKey: string,
  openAiKey: string,
) {
  const narration = narrationForDuration(
    String(job.narration_text || ""),
    Number(job.duration_seconds || 5),
  );
  const speechResponse = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${openAiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: Deno.env.get("OPENAI_TTS_MODEL") ?? "gpt-4o-mini-tts",
      voice: job.voice === "onyx" ? "onyx" : "nova",
      input: narration,
      response_format: "mp3",
      instructions: "Habla con claridad, calidez y ritmo natural. Termina dentro de la duración del video.",
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!speechResponse.ok) throw new Error("tts_failed");

  const bytes = new Uint8Array(await speechResponse.arrayBuffer());
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  const audioUrl = `data:audio/mpeg;base64,${btoa(binary)}`;

  const mergeResponse = await fetch(`https://queue.fal.run/${MERGE_MODEL}`, {
    method: "POST",
    headers: {
      "Authorization": `Key ${falKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      video_url: videoUrl,
      audio_url: audioUrl,
      start_offset: 0,
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!mergeResponse.ok) throw new Error("merge_start_failed");
  const queued = await mergeResponse.json();
  if (!queued.request_id || !queued.status_url || !queued.response_url) {
    throw new Error("merge_start_invalid");
  }
  queueUrl(queued.status_url);
  queueUrl(queued.response_url);
  return queued;
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
  const openAiKey = Deno.env.get("OPENAI_API_KEY");
  if (!url || !anonKey || !serviceKey || !falKey || !openAiKey) {
    return json(origin, {
      message: "El servicio de video con voz no está disponible en este momento.",
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

  const body = await readJsonRequest(request).catch(() => ({} as Record<string, unknown>));
  const jobId = typeof body.job_id === "string" ? body.job_id : "";
  if (!jobId) {
    return json(origin, { message: "Falta el identificador del video." }, 400);
  }

  const service = createClient(url, serviceKey);
  const { data: job, error: jobError } = await service.from("video_jobs").select("*")
    .eq("id", jobId).eq("user_id", userData.user.id).maybeSingle();
  if (jobError) return json(origin, { message: "No pudimos consultar el seguimiento del video." }, 503);
  if (!job) return json(origin, { message: "No encontramos ese video." }, 404);
  if (job.status === "completed") {
    return settleVideo(service, job, origin, true, job.video_url);
  }
  if (job.status === "failed") return settleVideo(service, job, origin, false, null, job.error_code);

  // A provider request can expire or stop returning a useful state. Never leave
  // the user trapped in an endless "in progress" loop: close stale jobs and
  // release their reserved credits.
  const createdAt = Date.parse(String(job.created_at || ""));
  const maxJobAgeMs = 20 * 60 * 1000;
  if (Number.isFinite(createdAt) && Date.now() - createdAt > maxJobAgeMs) {
    return settleVideo(service, job, origin, false, null, "job_timeout");
  }

  const pollingMerge = Boolean(job.merge_request_id);
  const statusUrl = pollingMerge ? job.merge_status_url : job.status_url;
  let providerStatus: Response;
  try {
    providerStatus = await fetch(queueUrl(statusUrl), {
      headers: { "Authorization": `Key ${falKey}` },
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    return json(origin, {
      status: "in_progress",
      message: pollingMerge
        ? "Estamos agregando la voz al video."
        : "El video sigue procesándose.",
    }, 202);
  }
  if (!providerStatus.ok) {
    return json(origin, { status: "in_progress", message: "El video sigue procesándose." }, 202);
  }

  const state = await providerStatus.json();
  const falStatus = String(state.status ?? "").toUpperCase();
  if (falStatus === "IN_QUEUE" || falStatus === "IN_PROGRESS") {
    await service.from("video_jobs").update({
      status: "in_progress",
      updated_at: new Date().toISOString(),
    }).eq("id", job.id).in("status", ["queued", "in_progress"]);
    return json(origin, {
      status: "in_progress",
      stage: pollingMerge ? "adding_audio" : "generating_video",
    }, 202);
  }

  if (["FAILED", "CANCELLED"].includes(falStatus)) {
    return settleVideo(service, job, origin, false, null, pollingMerge ? "audio_merge_failed" : falStatus.toLowerCase());
  }

  if (falStatus === "COMPLETED") {
    const responseUrl = pollingMerge ? job.merge_response_url : job.response_url;
    let result: unknown = null;
    try {
      const resultResponse = await fetch(queueUrl(responseUrl), {
        headers: { "Authorization": `Key ${falKey}` },
        signal: AbortSignal.timeout(15_000),
      });
      result = resultResponse.ok ? await resultResponse.json() : null;
    } catch {
      result = null;
    }
    const videoUrl = extractVideoUrl(result);
    if (!videoUrl) {
      return json(origin, { status: "in_progress", message: "Estamos preparando el archivo final." }, 202);
    }

    if (!pollingMerge) {
      const { data: claimed, error: claimError } = await service.from("video_jobs")
        .update({ merge_claimed_at: new Date().toISOString() })
        .eq("id", job.id).in("status", ["queued", "in_progress"]).is("merge_claimed_at", null).select("id");
      if (claimError) return json(origin, { message: "No pudimos consultar la etapa de audio." }, 503);
      if (!claimed?.length) return json(origin, { status: "in_progress", stage: "adding_audio" }, 202);
      try {
        const merge = await queueAudioMerge(job, videoUrl, falKey, openAiKey);
        const { error: updateError } = await service.from("video_jobs").update({
          raw_video_url: videoUrl,
          merge_request_id: merge.request_id,
          merge_status_url: merge.status_url,
          merge_response_url: merge.response_url,
          status: "in_progress",
          updated_at: new Date().toISOString(),
        }).eq("id", job.id).in("status", ["queued", "in_progress"]);
        if (updateError) throw new Error("audio_tracking_failed");
        return json(origin, {
          status: "in_progress",
          stage: "adding_audio",
          message: "El video está listo; ahora estamos agregando la voz.",
        }, 202);
      } catch (error) {
        return settleVideo(service, job, origin, false, null, String((error as Error).message || "audio_pipeline_failed"));
      }
    }

    return settleVideo(service, job, origin, true, videoUrl);
  }

  return json(origin, {
    status: "in_progress",
    message: "El video sigue procesándose.",
  }, 202);
});
