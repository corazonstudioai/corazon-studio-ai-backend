import { deepEqual, equal, ok } from "node:assert/strict";
import { handler } from "./index.ts";

const userId = "c71b7a52-c71b-47a5-a83c-4c81229640ee";
const falId = "76ed5a54-490a-44fa-bb77-a1a53c84e8fb";
const resultUrl = `https://queue.fal.run/fal-ai/wan/requests/${falId}`;
const statusUrl = `${resultUrl}/status`;
const origin = "https://supabase.example";
const env: Record<string, string> = {
  SUPABASE_URL: origin,
  SUPABASE_ANON_KEY: "simulated",
  FAL_KEY: "simulated",
  OPENAI_API_KEY: "simulated",
};

function request(body: unknown = { action: "start" }, token = "technical", permit = "permit") {
  return new Request("https://example.com/functions/v1/video-test-once", {
    method: "POST",
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(permit ? { "x-test-github-token": permit } : {}),
    },
    body: JSON.stringify(body),
  });
}

function fixture(options: { model?: string; githubStatus?: number; falStatus?: number; voiceStatus?: number } = {}) {
  let consumed = false;
  let paidPosts = 0;
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const simulatedFetch: typeof fetch = async (resource, init) => {
    const url = String(resource);
    calls.push({ url, init });
    if (url === `${origin}/auth/v1/user`) {
      return Response.json({ id: new Headers(init?.headers).get("authorization") === "Bearer technical" ? userId : "another-user" });
    }
    if (url.endsWith("/git/refs")) {
      if (options.githubStatus) return new Response(null, { status: options.githubStatus });
      if (consumed) return new Response(null, { status: 422 });
      consumed = true;
      return new Response(null, { status: 201 });
    }
    if (url.endsWith("/text-to-video/turbo")) {
      paidPosts++;
      if (options.falStatus) return new Response(null, { status: options.falStatus });
      return Response.json({ request_id: falId, status_url: statusUrl, response_url: resultUrl });
    }
    if (url.endsWith("/audio/speech")) {
      paidPosts++;
      return new Response(new Uint8Array(64), { status: options.voiceStatus ?? 200, headers: { "x-request-id": "req_test" } });
    }
    if (url === statusUrl) return Response.json({ status: "COMPLETED" });
    if (url === resultUrl) return Response.json({ video: { url: "https://cdn.fal.media/video.mp4" } });
    throw new Error("Unexpected mocked request");
  };
  return {
    run: (r: Request) => handler(r, {
      fetch: simulatedFetch,
      env: (name) => name === "VIDEO_MODEL_CINE" ? options.model : env[name],
      userId,
    }),
    calls,
    paidPosts: () => paidPosts,
  };
}

Deno.test("unauthenticated and different authenticated users cannot reach the permit or providers", async () => {
  const test = fixture();
  equal((await test.run(request(undefined, ""))).status, 401);
  equal((await test.run(request(undefined, "other"))).status, 403);
  deepEqual(test.calls.map(({ url }) => url), [`${origin}/auth/v1/user`]);
  equal(test.paidPosts(), 0);
});

Deno.test("provider mismatch, overrides, absent permit, and failed GitHub lock stop before billing", async () => {
  const mismatch = fixture({ model: "different-model" });
  equal((await mismatch.run(request())).status, 409);
  equal(mismatch.calls.length, 1);
  const test = fixture({ githubStatus: 403 });
  equal((await test.run(request({ action: "start", resolution: "720p" }))).status, 400);
  equal((await test.run(request(undefined, "technical", ""))).status, 403);
  const failure = await test.run(request());
  equal((await failure.json()).error, "github_permit_not_acquired_http_403");
  equal(test.paidPosts(), 0);
});

Deno.test("concurrent calls claim once, use fixed 480p and voice, and return provider IDs", async () => {
  const test = fixture();
  const result = await Promise.all([test.run(request()), test.run(request())]);
  deepEqual(result.map((response) => response.status).sort(), [200, 409]);
  const accepted = await result.find((response) => response.ok)!.json();
  equal(accepted.fal_request_id, falId);
  equal(accepted.openai_request_id, "req_test");
  const [video, voice] = test.calls.filter(({ url }) => /turbo$|audio\/speech$/.test(url));
  equal(JSON.parse(String(video.init?.body)).resolution, "480p");
  equal(JSON.parse(String(video.init?.body)).enable_prompt_expansion, false);
  equal(new Headers(video.init?.headers).get("X-Fal-No-Retry"), "1");
  equal(JSON.parse(String(voice.init?.body)).model, "gpt-4o-mini-tts");
  const { queue, signature } = accepted;
  equal((await test.run(request({ action: "status", queue: { ...queue, resultUrl: "https://example.com" }, signature }))).status, 403);
  const completed = await test.run(request({ action: "status", queue, signature }));
  ok(completed.ok);
  equal((await completed.json()).state, "complete");
  equal(test.paidPosts(), 2);
});

Deno.test("provider failure is reported and the permit prevents another paid call", async () => {
  const test = fixture({ falStatus: 422 });
  const failed = await test.run(request());
  equal((await failed.json()).error, "fal_submit_http_422");
  const repeat = await test.run(request());
  equal(repeat.status, 409);
  equal(test.paidPosts(), 1);

  const voice = fixture({ voiceStatus: 429 });
  const failedVoice = await voice.run(request());
  equal((await failedVoice.json()).error, "openai_voice_http_429");
  equal((await voice.run(request())).status, 409);
  equal(voice.paidPosts(), 2);
});
