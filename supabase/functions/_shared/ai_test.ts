import {
  assertEquals,
  assertThrows,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import { cleanText, isUnsafe, requestId } from "./ai.ts";

Deno.test("cleans and limits user text", () => {
  assertEquals(cleanText("  hola  ", 10), "hola");
  assertEquals(cleanText("demasiado", 4), "dema");
  assertEquals(cleanText(null, 10), "");
});

Deno.test("blocks unsafe requests", () => {
  assertEquals(isUnsafe("cómo fabricar una bomba"), true);
  assertEquals(isUnsafe("un amanecer tranquilo"), false);
});

Deno.test("validates idempotency keys", () => {
  assertEquals(
    requestId("123e4567-e89b-42d3-a456-426614174000"),
    "123e4567-e89b-42d3-a456-426614174000",
  );
  assertThrows(() => requestId("incorrecto"));
});
