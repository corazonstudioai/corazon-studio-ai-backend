import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "./video.ts";

export function json(origin: string | null, body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders(origin),
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}

export function cleanText(value: unknown, max: number) {
  return typeof value === "string" ? value.trim().slice(0, max) : "";
}

export function isUnsafe(text: string) {
  const normalized = text.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  const blocked = [
    /\b(?:porn\w*|porno\w*|nude|nudity|desnud[oa]\w*|gore|decapitat\w*|bestial\w*)/i,
    /(?:sexo\s+explicito|explicit\s+sex|sexual\s+explicit)/i,
    /sexual\s+(?:con|de)\s+menores/i,
    /pornograf/i,
    /desnud[oa]s?\s+(?:infantil|menor|niñ)/i,
    /abuso\s+sexual/i,
    /como\s+(?:matar|fabricar\s+una\s+bomba)/i,
  ];
  return blocked.some((rule) => rule.test(normalized));
}

export function requestId(value: unknown) {
  const candidate = typeof value === "string" ? value : crypto.randomUUID();
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
      .test(candidate)
  ) {
    throw new Error("invalid_idempotency_key");
  }
  return candidate;
}

export async function hashRequest(body: Record<string, unknown>) {
  const normalized = Object.fromEntries(Object.keys(body).filter(key => key !== "idempotency_key").sort().map(key => [key, body[key]]));
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(normalized)));
  return [...new Uint8Array(bytes)].map(byte => byte.toString(16).padStart(2,"0")).join("");
}

type ReservationOptions = {
  resource: "text_requests" | "images" | "voice_minutes";
  units: number;
  creditCost: number;
  estimatedCost: number;
  engine: string;
  idempotencyKey: string;
  requestHash: string;
};

export async function authenticateAndReserve(
  request: Request,
  options: ReservationOptions,
) {
  const url = Deno.env.get("SUPABASE_URL");
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !anonKey || !serviceKey) throw new Error("service_unavailable");

  const auth = request.headers.get("authorization") ?? "";
  const userClient = createClient(url, anonKey, {
    global: { headers: { Authorization: auth } },
  });
  const { data, error } = await userClient.auth.getUser();
  const user = data.user;
  if (error || !user) throw new Error("unauthorized");
  const identityKind = user.email_confirmed_at
    ? "email"
    : user.phone_confirmed_at
    ? "phone"
    : null;
  if (!identityKind) throw new Error("identity_not_verified");

  const service = createClient(url, serviceKey);
  const { data: rows, error: reserveError } = await service.rpc(
    "reserve_generation_v2",
    {
      p_user_id: user.id,
      p_identity_kind: identityKind,
      p_resource: options.resource,
      p_units: options.units,
      p_credit_cost: options.creditCost,
      p_estimated_cost_usd: options.estimatedCost,
      p_engine: options.engine,
      p_idempotency_key: options.idempotencyKey,
      p_request_hash: options.requestHash,
    },
  );
  const reservation = rows?.[0];
  if (reserveError || !reservation?.allowed) {
    const failure = new Error(reservation?.reason ?? "reservation_failed");
    (failure as Error & { creditsRemaining?: number }).creditsRemaining =
      reservation?.credits_remaining;
    throw failure;
  }
  return { service, reservation };
}

export async function finalize(
  service: unknown,
  reservationId: string,
  success: boolean,
) {
  const client = service as {
    rpc: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  };

  const result = await client.rpc("finalize_generation", {
    p_reservation_id: reservationId,
    p_success: success,
  });
  if ((result as { error?: unknown })?.error) throw new Error("billing_unavailable");
}

export function failureResponse(origin: string | null, error: unknown) {
  const code = String((error as Error)?.message ?? error);
  if (code === "request_too_large") return json(origin, { code, message: "La solicitud es demasiado larga. Acórtala y vuelve a intentar." }, 413);
  if (["generation_in_progress", "generation_already_refunded", "idempotency_conflict"].includes(code)) {
    return json(origin, { code, message: code === "generation_in_progress"
      ? "Tu solicitud sigue procesándose. No necesitas crearla otra vez."
      : "Esta solicitud ya fue cerrada o cambió. Inicia una creación nueva." }, 409);
  }
  if (["invalid_brain_plan", "unsupported_brain_plan"].includes(code)) {
    return json(origin, { code, message: "No pudimos preparar un plan completo para esta solicitud. No se inició la generación de archivos." }, 502);
  }
  if (
    code === "empty_speech" || code === "speech_too_long" ||
    code === "invalid_request"
  ) {
    return json(origin, {
      code,
      message: code === "speech_too_long"
        ? "El texto de voz admite hasta 1200 caracteres. Acórtalo e inténtalo nuevamente."
        : "Escribe el texto que deseas escuchar.",
    }, 400);
  }
  if (code === "unauthorized") {
    return json(origin, { message: "Inicia sesión para continuar." }, 401);
  }
  if (code === "identity_not_verified") {
    return json(origin, {
      message: "Verifica tu correo o teléfono para continuar.",
    }, 403);
  }
  if (code === "credits_exhausted") {
    return json(origin, { message: "Tus créditos se terminaron." }, 402);
  }
  if (code === "resource_limit_reached") {
    return json(origin, {
      message: "Llegaste al límite de esta función en tu plan.",
    }, 403);
  }
  if (code === "service_unavailable") {
    return json(origin, {
      message: "El servicio no está configurado en este momento.",
    }, 503);
  }
  return json(origin, {
    message: "No pudimos completar la solicitud en este momento.",
  }, 500);
}

export async function openAi(path: string, init: RequestInit) {
  const apiKey = Deno.env.get("OPENAI_API_KEY");
  if (!apiKey) throw new Error("service_unavailable");
  return fetch(`https://api.openai.com/v1/${path}`, {
    ...init,
    signal: init.signal ?? AbortSignal.timeout(path === "images/generations" ? 80_000 : 45_000),
    headers: {
      Authorization: `Bearer ${apiKey}`,
      ...(init.headers ?? {}),
    },
  });
}

export function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}
