import { assertEquals, assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { extractVideoUrl, parseVideoRequest } from "./video.ts";

Deno.test("validates video request", () => {
  const id = "123e4567-e89b-42d3-a456-426614174000";
  assertEquals(parseVideoRequest({ prompt: "Un amanecer", duration: 5, idempotency_key: id }), {
    prompt: "Un amanecer",
    duration: 5,
    idempotencyKey: id,
  });
  assertThrows(() => parseVideoRequest({ prompt: "x", duration: 5 }));
  assertThrows(() => parseVideoRequest({ prompt: "válido", duration: 7 }));
});

Deno.test("extracts fal video response", () => {
  assertEquals(extractVideoUrl({ video: { url: "https://example.com/video.mp4" } }), "https://example.com/video.mp4");
  assertEquals(extractVideoUrl({ videos: [{ url: "https://example.com/other.mp4" }] }), "https://example.com/other.mp4");
  assertEquals(extractVideoUrl({}), null);
});
