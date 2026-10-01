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
  const blocked = [
    /sexual\s+(?:con|de)\s+menores/i,
    /pornograf/i,
    /desnud[oa]s?\s+(?:infantil|menor|niñ)/i,
    /abuso\s+sexual/i,
    /cómo\s+(?:matar|fabricar\s+una\s+bomba)/i,
  ];
  return blocked.some((rule) => rule.test(text));
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

type ReservationOptions = {
  resource: "text_requests" | "images" | "voice_minutes";
  units: number;
  creditCost: number;
  estimatedCost: number;
  engine: string;
  idempotencyKey: string;
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
    "reserve_generation",
    {
      p_user_id: user.id,
      p_identity_kind: identityKind,
      p_resource: options.resource,
      p_units: options.units,
      p_credit_cost: options.creditCost,
      p_estimated_cost_usd: options.estimatedCost,
      p_engine: options.engine,
      p_idempotency_key: options.idempotencyKey,
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
  service: ReturnType<typeof createClient>,
  reservationId: string,
  success: boolean,
) {
  const rpc = service.rpc as unknown as (
    name: string,
    args: Record<string, unknown>,
  ) => Promise<unknown>;

  await rpc("finalize_generation", {
    p_reservation_id: reservationId,
    p_success: success,
  });
}

export function failureResponse(origin: string | null, error: unknown) {
  const code = String((error as Error)?.message ?? error);
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
