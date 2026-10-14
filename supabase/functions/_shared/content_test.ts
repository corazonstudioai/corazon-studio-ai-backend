import {
  assertEquals,
  assertThrows,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  languageInstruction,
  parseSpeechRequest,
  requireReply,
} from "./content.ts";

Deno.test("explicit output language overrides input language", () => {
  assertEquals(languageInstruction("en"), "Write the result in English.");
  assertEquals(languageInstruction("es"), "Escribe el resultado en español.");
  assertEquals(languageInstruction("fr"), "Responde en el idioma del usuario.");
});

Deno.test("speech preserves spoken text and keeps style separate", () => {
  const speech = parseSpeechRequest({
    text: "  Dios va adelante.  ",
    voice: "onyx",
    instructions: "Habla con calidez.",
  });
  assertEquals(speech, {
    text: "Dios va adelante.",
    voice: "onyx",
    instructions: "Habla con calidez.",
  });
  assertEquals(
    parseSpeechRequest({ text: "Hello", voice: "invalid" }).voice,
    "nova",
  );
});

Deno.test("speech rejects long input instead of silently truncating", () => {
  assertThrows(
    () => parseSpeechRequest({ text: "a".repeat(1201) }),
    Error,
    "speech_too_long",
  );
  assertEquals(
    parseSpeechRequest({ text: "a".repeat(1200) }).text.length,
    1200,
  );
  assertThrows(() => parseSpeechRequest({ text: " " }), Error, "empty_speech");
});

Deno.test("empty provider replies cannot be charged as successful", () => {
  assertThrows(() => requireReply("  "), Error, "provider_error");
  assertThrows(() => requireReply(null), Error, "provider_error");
  assertEquals(requireReply(" Hola "), "Hola");
});
