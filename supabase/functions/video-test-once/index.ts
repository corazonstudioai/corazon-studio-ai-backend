const technicalUserId = "__VIDEO_TEST_USER_ID__";
const model = "fal-ai/wan/v2.2-a14b/text-to-video/turbo";
const voiceModel = "gpt-4o-mini-tts";
const repo = "corazonstudioai/corazon-studio-ai-backend";
const permitRef = "refs/tags/controlled-video-test-consumed-v1";
const permitCommit = "73ae34fc9e171697781137d3466a23aa02ea2236";
const uuid = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i;
const encoder = new TextEncoder();

type Queue = { id: string; statusUrl: string; resultUrl: string };

function reply(status: number, body: unknown) {
  return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
}

function evidence(provider: string, id: string | null, status: number) {
  console.info(JSON.stringify({ event: "controlled_video_test", provider, request_id: id, http_status: status }));
}

function queueUrl(value: string, id: string, suffix: string) {
  const url = new URL(value);
  if (url.origin !== "https://queue.fal.run" || url.username || url.password ||
      url.search || url.hash || !url.pathname.endsWith(`/requests/${id}${suffix}`)) {
    throw new Error("fal_invalid_queue_url");
  }
  return url.href;
}

async function signingKey(key: string) {
  return await crypto.subtle.importKey(
    "raw", encoder.encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"],
  );
}

function signatureInput(queue: Queue) {
  return encoder.encode(JSON.stringify([permitRef, technicalUserId, queue.id, queue.statusUrl, queue.resultUrl]));
}

function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

export function handler(
  request: Request,
  deps = { fetch: globalThis.fetch, env: Deno.env.get.bind(Deno.env), userId: technicalUserId },
): Promise<Response> {
  return handle(request, deps);
}

async function handle(request: Request, deps: { fetch: typeof fetch; env: (name: string) => string | undefined; userId: string }) {
  const send = (url: string, init: RequestInit = {}) =>
    deps.fetch(url, { ...init, redirect: "error", signal: AbortSignal.timeout(25_000) });

  const authorization = request.headers.get("authorization");
  if (!authorization?.startsWith("Bearer ")) return reply(401, { error: "authentication_required" });
  if (!uuid.test(deps.userId)) return reply(503, { error: "technical_account_not_configured" });

  const project = deps.env("SUPABASE_URL");
  const anonKey = deps.env("SUPABASE_ANON_KEY");
  if (!project || !anonKey) return reply(503, { error: "supabase_auth_not_configured" });
  let stage = "supabase_auth";
  let falId: string | null = null;
  let voiceId: string | null = null;
  try {
    const auth = await send(`${project}/auth/v1/user`, { headers: { authorization, apikey: anonKey } });
    if (!auth.ok) return reply(auth.status >= 500 ? 503 : 401, { error: `supabase_auth_http_${auth.status}` });
    const user = await auth.json();
    if (user.id !== deps.userId) return reply(403, { error: "technical_account_required" });

    if (request.method !== "POST") return reply(405, { error: "method_not_allowed" });
    stage = "invalid_request";
    const body = await request.json();
    const falKey = deps.env("FAL_KEY");
    const openAiKey = deps.env("OPENAI_API_KEY");
    if (!falKey || !openAiKey) return reply(503, { error: "provider_credentials_missing" });
    if ((deps.env("VIDEO_MODEL_CINE") ?? model) !== model) {
      return reply(409, { error: "video_test_model_mismatch" });
    }
    if ((deps.env("OPENAI_TTS_MODEL") ?? voiceModel) !== voiceModel) {
      return reply(409, { error: "video_test_voice_mismatch" });
    }
    const falAuth = { Authorization: `Key ${falKey}` };

    if (body?.action === "status") {
      stage = "fal_invalid_continuation";
      const queue: Queue = body.queue;
      if (!queue || !uuid.test(queue.id) || typeof body.signature !== "string" || body.signature.length !== 44) {
        return reply(400, { error: stage });
      }
      const valid = await crypto.subtle.verify(
        "HMAC", await signingKey(falKey),
        Uint8Array.from(atob(body.signature), (character) => character.charCodeAt(0)),
        signatureInput(queue),
      );
      if (!valid) return reply(403, { error: stage });
      falId = queue.id;
      stage = "fal_status";
      const status = await send(queueUrl(queue.statusUrl, falId, "/status"), { headers: falAuth });
      if (!status.ok) return reply(502, { error: `fal_status_http_${status.status}`, fal_request_id: falId });
      const state = await status.json();
      if (state.status === "IN_QUEUE" || state.status === "IN_PROGRESS") return reply(200, { state: "pending" });
      if (state.status !== "COMPLETED" || state.error) {
        const providerCode = /^[a-z0-9_]{1,80}$/i.test(state.error_type ?? "") ? state.error_type : "job_failed";
        return reply(502, { error: `fal_${providerCode}`, fal_request_id: falId });
      }
      stage = "fal_result";
      const result = await send(queueUrl(queue.resultUrl, falId, ""), { headers: falAuth });
      if (!result.ok) return reply(502, { error: `fal_result_http_${result.status}`, fal_request_id: falId });
      const media = new URL((await result.json()).video.url);
      if (media.protocol !== "https:" || media.username || media.password || media.port ||
          !(media.hostname.endsWith(".fal.media") || media.hostname === "storage.googleapis.com")) {
        return reply(502, { error: "fal_invalid_video_url", fal_request_id: falId });
      }
      return reply(200, { state: "complete", video_url: media.href, fal_request_id: falId });
    }

    if (body?.action !== "start" || Object.keys(body).length !== 1) {
      return reply(400, { error: "invalid_request" });
    }
    const permitToken = request.headers.get("x-test-github-token");
    if (!permitToken) return reply(403, { error: "github_permit_required" });
    // A Git ref can be created only once. Claim it before either paid POST.
    stage = "github_permit";
    const claim = await send(`https://api.github.com/repos/${repo}/git/refs`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${permitToken}`,
        Accept: "application/vnd.github+json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: permitRef, sha: permitCommit }),
    });
    if (claim.status !== 201) {
      return reply(409, { error: `github_permit_not_acquired_http_${claim.status}` });
    }

    stage = "fal_submit";
    const fal = await send(`https://queue.fal.run/${model}`, {
      method: "POST",
      headers: {
        ...falAuth,
        "Content-Type": "application/json",
        "X-Fal-No-Retry": "1",
        "x-app-fal-disable-fallback": "true",
      },
      body: JSON.stringify({
        prompt: "Un amanecer sobre montañas verdes. Las nubes avanzan lentamente y la cámara se mueve con suavidad. Sin texto.",
        resolution: "480p",
        aspect_ratio: "16:9",
        enable_prompt_expansion: false,
        enable_safety_checker: true,
        enable_output_safety_checker: true,
      }),
    });
    falId = fal.headers.get("x-fal-request-id");
    if (!fal.ok) {
      evidence("fal", falId, fal.status);
      return reply(502, { error: `fal_submit_http_${fal.status}`, fal_request_id: falId });
    }
    const queued = await fal.json();
    if (!uuid.test(queued.request_id)) return reply(502, { error: "fal_invalid_request_id", fal_request_id: falId });
    falId = queued.request_id;
    evidence("fal", falId, fal.status);
    const queue = {
      id: falId!,
      statusUrl: queueUrl(queued.status_url, falId!, "/status"),
      resultUrl: queueUrl(queued.response_url, falId!, ""),
    };
    const signature = bytesToBase64(new Uint8Array(
      await crypto.subtle.sign("HMAC", await signingKey(falKey), signatureInput(queue)),
    ));

    stage = "openai_voice";
    const voice = await send("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: { Authorization: `Bearer ${openAiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: voiceModel,
        voice: "nova",
        input: "Dios va delante de ti.",
        instructions: "Habla en español, con claridad y calidez. Termina antes de cinco segundos.",
        response_format: "wav",
      }),
    });
    voiceId = voice.headers.get("x-request-id");
    evidence("openai", voiceId, voice.status);
    if (!voice.ok) return reply(502, { error: `openai_voice_http_${voice.status}`, fal_request_id: falId, openai_request_id: voiceId });
    const audio = new Uint8Array(await voice.arrayBuffer());
    if (audio.byteLength < 44 || audio.byteLength > 1_000_000) {
      return reply(502, { error: "openai_voice_invalid_size", fal_request_id: falId, openai_request_id: voiceId });
    }
    return reply(200, { queue, signature, audio_base64: bytesToBase64(audio), fal_request_id: falId, openai_request_id: voiceId });
  } catch (error) {
    const suffix = error instanceof Error && error.name === "TimeoutError" ? "timeout" : "invalid_response_or_connection";
    return reply(stage === "invalid_request" ? 400 : 502, {
      error: `${stage}_${suffix}`, fal_request_id: falId, openai_request_id: voiceId,
    });
  }
}

if (import.meta.main) Deno.serve((request) => handler(request));
