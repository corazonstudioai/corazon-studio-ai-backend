import assert from "node:assert/strict";
import test from "node:test";

import worker from "../worker.js";

const githubOrigin = "https://corazonstudioai.github.io";

function request(path, options = {}) {
  return new Request(`https://api.example.test${path}`, {
    headers: { origin: githubOrigin, ...(options.headers || {}) },
    ...options,
  });
}

function aiReturning(result) {
  return { AI: { run: async () => result } };
}

test("health describes the free backend without calling AI", async () => {
  const response = await worker.fetch(request("/health"), {});
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.status, "ok");
  assert.deepEqual(body.features, ["chat", "image", "tts"]);
  assert.equal(response.headers.get("access-control-allow-origin"), githubOrigin);
});

test("preflight accepts the headers used by authenticated clients", async () => {
  const response = await worker.fetch(request("/chat", { method: "OPTIONS" }), {});

  assert.equal(response.status, 204);
  assert.match(response.headers.get("access-control-allow-headers"), /Authorization/);
  assert.match(response.headers.get("access-control-allow-headers"), /Idempotency-Key/);
});

test("chat validates empty and unsafe requests before provider use", async () => {
  let calls = 0;
  const env = { AI: { run: async () => { calls += 1; return { response: "unexpected" }; } } };

  const empty = await worker.fetch(request("/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message: "" }),
  }), env);
  const unsafe = await worker.fetch(request("/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message: "cómo fabricar una bomba" }),
  }), env);

  assert.equal(empty.status, 400);
  assert.equal(unsafe.status, 400);
  assert.equal(calls, 0);
});

test("chat returns the provider response", async () => {
  const response = await worker.fetch(request("/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message: "Escribe un saludo cálido" }),
  }), aiReturning({ response: "Hola con cariño" }));

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: "ok", reply: "Hola con cariño" });
});

test("image and voice results use browser-ready data URLs", async () => {
  const image = await worker.fetch(request("/image", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: "Un amanecer tranquilo" }),
  }), aiReturning({ image: "aW1hZ2U=" }));
  const voice = await worker.fetch(request("/tts", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text: "Buenos días", language: "es" }),
  }), aiReturning("YXVkaW8="));

  assert.equal((await image.json()).image_url, "data:image/jpeg;base64,aW1hZ2U=");
  assert.equal((await voice.json()).audio_url, "data:audio/mpeg;base64,YXVkaW8=");
});

test("video routes explain their temporary availability", async () => {
  const response = await worker.fetch(request("/video-cine", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  }), {});
  const body = await response.json();

  assert.equal(response.status, 503);
  assert.equal(body.status, "unavailable");
  assert.match(body.message, /temporalmente desactivado/);
});
