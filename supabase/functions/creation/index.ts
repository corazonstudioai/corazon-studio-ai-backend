import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { json } from "../_shared/ai.ts";
import { corsHeaders } from "../_shared/video.ts";
import { readJsonRequest } from "../_shared/request.ts";
import { hydrateResult } from "../_shared/results.ts";

// Read completed work only. This endpoint never reserves credits, settles jobs,
// or calls a generation provider, even when a reference is missing or expired.
Deno.serve(async request => {
  const origin = request.headers.get("origin");
  const reply = (body: unknown, status = 200) => {
    const response = json(origin, body, status);
    response.headers.set("Cache-Control", "no-store");
    return response;
  };
  if (request.method === "OPTIONS") return new Response("ok", {headers: corsHeaders(origin)});
  if (request.method !== "POST") return reply({code: "method_not_allowed"}, 405);
  try {
    const body = await readJsonRequest(request);
    if (!(["generation", "video"].includes(String(body.type))) || typeof body.type !== "string" ||
      typeof body.id !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(body.id)) {
      return reply({code: "invalid_creation_reference"}, 400);
    }
    const url = Deno.env.get("SUPABASE_URL"), anon = Deno.env.get("SUPABASE_ANON_KEY"), key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!url || !anon || !key) return reply({code: "creation_unavailable"}, 503);
    const client = createClient(url, anon, {global: {headers: {Authorization: request.headers.get("authorization") || ""}}});
    const {data: auth, error: authError} = await client.auth.getUser();
    if (authError || !auth?.user) return reply({code: "unauthorized"}, 401);
    const service = createClient(url, key);
    if (body.type === "video") {
      const {data: job, error} = await service.from("video_jobs").select("status,video_url")
        .eq("id", body.id).eq("user_id", auth.user.id).maybeSingle();
      if (error) return reply({code: "creation_unavailable"}, 503);
      if (!job) return reply({code: "creation_not_found"}, 404);
      if (job.status !== "completed") return reply({code: "creation_not_ready"}, 409);
      if (typeof job.video_url !== "string" || !job.video_url.startsWith("https://")) return reply({code: "creation_unavailable"}, 503);
      return reply({mode: "video", data: {video_url: job.video_url}});
    }
    const {data: event, error} = await service.from("generation_events").select("id,status,resource")
      .eq("idempotency_key", body.id).eq("user_id", auth.user.id).maybeSingle();
    if (error) return reply({code: "creation_unavailable"}, 503);
    if (!event) return reply({code: "creation_not_found"}, 404);
    if (event.status !== "succeeded") return reply({code: "creation_not_ready"}, 409);
    const mode = ({text_requests: "text", images: "image", voice_minutes: "voice"} as Record<string, string>)[event.resource];
    if (!mode) return reply({code: "creation_not_found"}, 404);
    const saved = await service.from("generation_results").select("result").eq("reservation_id", event.id).maybeSingle();
    if (saved.error) return reply({code: "creation_unavailable"}, 503);
    if (!saved.data?.result) return reply({code: "creation_not_found"}, 404);
    const hydrated = await hydrateResult(service, saved.data.result);
    const field = mode === "text" ? "reply" : mode === "image" ? "image_url" : "audio_url";
    if (typeof hydrated[field] !== "string" || !hydrated[field]) return reply({code: "creation_unavailable"}, 503);
    return reply({mode, data: {[field]: hydrated[field]}});
  } catch (error) {
    return reply({code: error instanceof Error && ["invalid_request", "request_too_large"].includes(error.message) ? "invalid_creation_reference" : "creation_unavailable"},
      error instanceof Error && ["invalid_request", "request_too_large"].includes(error.message) ? 400 : 503);
  }
});
