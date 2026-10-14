import {
  assertEquals,
  assertThrows,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  corsHeaders,
  extractVideoUrl,
  narrationForDuration,
  parseVideoRequest,
  silentVideoTimeline,
  videoAudioTimeline,
  pcmSpeechAudio,
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
      audioMode: "narrated",
      aspectRatio: "16:9",
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
  assertThrows(()=>narrationForDuration("uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece",5));
  assertEquals(narrationForDuration("Mensaje breve", 10), "Mensaje breve");
});

Deno.test("composition preserves the visual timeline with short narration", () => {
  const body=videoAudioTimeline("https://example.com/visual.mp4","data:audio/mpeg;base64,YQ==",5,2);
  assertEquals(body.tracks[0].type,"video");
  assertEquals(body.tracks[0].keyframes[0].duration,5000);
  assertEquals(body.tracks[1].keyframes[0].timestamp,0);
  assertEquals(body.tracks[1].keyframes[0].url,"data:audio/mpeg;base64,YQ==");
  assertEquals(videoAudioTimeline("video","audio",10,2).tracks[0].keyframes[0].duration,10000);
  assertThrows(()=>videoAudioTimeline("video","audio",7,2));
  assertEquals(extractVideoUrl({video_url:"https://example.com/compose.mp4"}),"https://example.com/compose.mp4");
});

Deno.test("PCM speech keeps its actual length and cannot loop or truncate",()=>{
  const pcm=new Uint8Array(48000);pcm[0]=1;const audio=pcmSpeechAudio(pcm);
  assertEquals(audio.durationSeconds,1);assertEquals(audio.wav.byteLength,48044);assertEquals(audio.wav[44],1);
  const view=new DataView(audio.wav.buffer);assertEquals(view.getUint32(24,true),24000);assertEquals(view.getUint16(34,true),16);
  const tracks=videoAudioTimeline("video","audio",5,audio.durationSeconds).tracks;
  assertEquals(tracks[0].keyframes[0].duration,5000);assertEquals(tracks[1].keyframes[0].duration,1000);
  assertThrows(()=>pcmSpeechAudio(new Uint8Array()));assertThrows(()=>pcmSpeechAudio(new Uint8Array(3)));
  assertThrows(()=>videoAudioTimeline("video","audio",5,6));assertThrows(()=>videoAudioTimeline("video","audio",5,NaN));
});

Deno.test("silent requests are explicit and cannot carry conflicting narration",()=>{
  const base={prompt:"A moving sunrise",duration:5,audio_mode:"silent"};
  assertEquals(parseVideoRequest({...base,narration:""}).audioMode,"silent");
  assertEquals(parseVideoRequest(base).narration,"");
  assertThrows(()=>parseVideoRequest({...base,narration:"Speak this"}));
  assertThrows(()=>parseVideoRequest({...base,audio_mode:null}));
  assertThrows(()=>parseVideoRequest({...base,audio_mode:"unknown"}));
  assertEquals(silentVideoTimeline("https://example.com/video.mp4",5).tracks.length,1);
  assertThrows(()=>silentVideoTimeline("video",7));
});

Deno.test("video aspect ratios are bounded and legacy defaults remain horizontal",()=>{
  for(const aspect_ratio of ["9:16","16:9","1:1"])assertEquals(parseVideoRequest({prompt:"A sunrise",aspect_ratio}).aspectRatio,aspect_ratio);
  assertThrows(()=>parseVideoRequest({prompt:"A sunrise",aspect_ratio:"4:3"}));
});
