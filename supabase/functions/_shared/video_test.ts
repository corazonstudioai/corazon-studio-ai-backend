import {
  assertEquals,
  assertThrows,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  corsHeaders,
  extractVideoUrl,
  narrationForDuration,
  parseVideoRequest,
  videoAudioTimeline,
} from "./video.ts";

Deno.test("validates video request", () => {
  const id = "123e4567-e89b-42d3-a456-426614174000";
  assertEquals(
    parseVideoRequest({
      prompt: "Un amanecer",
      duration: 5,
      idempotency_key: id,
    }),
    {
      prompt: "Un amanecer",
      duration: 5,
      idempotencyKey: id,
      narration: "Un amanecer",
      voice: "nova",
    },
  );
  assertThrows(() => parseVideoRequest({ prompt: "x", duration: 5 }));
  assertThrows(() => parseVideoRequest({ prompt: "válido", duration: 7 }));
});

Deno.test("extracts fal video response", () => {
  assertEquals(
    extractVideoUrl({ video: { url: "https://example.com/video.mp4" } }),
    "https://example.com/video.mp4",
  );
  assertEquals(
    extractVideoUrl({ videos: [{ url: "https://example.com/other.mp4" }] }),
    "https://example.com/other.mp4",
  );
  assertEquals(extractVideoUrl({}), null);
});


Deno.test("advertises every supported CORS method", () => {
  assertEquals(
    corsHeaders("https://corazonstudioai.github.io")["Access-Control-Allow-Methods"],
    "GET, POST, OPTIONS",
  );
});

Deno.test("prepares narration that fits the video duration", () => {
  assertEquals(
    narrationForDuration("uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece", 5),
    "uno dos tres cuatro cinco seis siete ocho nueve diez once doce…",
  );
  assertEquals(narrationForDuration("Mensaje breve", 10), "Mensaje breve");
});

Deno.test("composition preserves the visual timeline with short narration", () => {
  const body=videoAudioTimeline("https://example.com/visual.mp4","data:audio/mpeg;base64,YQ==",5);
  assertEquals(body.tracks[0].type,"video");
  assertEquals(body.tracks[0].keyframes[0].duration,5000);
  assertEquals(body.tracks[1].keyframes[0].timestamp,0);
  assertEquals(body.tracks[1].keyframes[0].url,"data:audio/mpeg;base64,YQ==");
  assertEquals(videoAudioTimeline("video","audio",10).tracks[0].keyframes[0].duration,10000);
  assertThrows(()=>videoAudioTimeline("video","audio",7));
  assertEquals(extractVideoUrl({video_url:"https://example.com/compose.mp4"}),"https://example.com/compose.mp4");
});
