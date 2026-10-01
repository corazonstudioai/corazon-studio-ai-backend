import { corsHeaders } from "../_shared/video.ts";
import { json } from "../_shared/ai.ts";

Deno.serve((request) => {
  const origin = request.headers.get("origin");
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders(origin) });
  }
  if (request.method !== "GET" && request.method !== "POST") {
    return json(origin, { message: "Método no permitido." }, 405);
  }
  return json(origin, {
    status: "ok",
    message: "Corazón Studio AI activo",
    platform: "Supabase Edge Functions",
    features: ["chat", "image", "tts", "video"],
  });
});
