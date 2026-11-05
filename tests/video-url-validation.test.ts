import { extractVideoUrl, safeVideoUrl } from "../supabase/functions/_shared/video.ts";

Deno.test("only HTTPS video links without embedded credentials are accepted", () => {
  const valid = "https://cdn.example.org/video.mp4";
  if (safeVideoUrl(valid) !== valid) throw new Error("HTTPS video rejected");
  for (const candidate of [
    "http://cdn.example.org/video.mp4",
    "javascript:alert(1)",
    "data:video/mp4;base64,AAAA",
    "https://user:password@cdn.example.org/video.mp4",
    "https://localhost/video.mp4",
    "not a URL",
    "",
    null,
  ]) {
    if (safeVideoUrl(candidate) !== null) throw new Error("Unsafe video URL accepted");
  }
});

Deno.test("provider response parser rejects unsafe URLs and accepts supported shapes", () => {
  if (extractVideoUrl({ video: { url: "https://cdn.example.org/a.mp4" } }) !== "https://cdn.example.org/a.mp4") {
    throw new Error("Nested video URL not read");
  }
  if (extractVideoUrl({ videos: [{ url: "https://cdn.example.org/b.mp4" }] }) !== "https://cdn.example.org/b.mp4") {
    throw new Error("Video list URL not read");
  }
  if (extractVideoUrl({ video_url: "http://invalid.example.org/a.mp4" }) !== null) {
    throw new Error("Unsafe video incorrectly accepted");
  }
});
