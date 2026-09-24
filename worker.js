const ALLOWED_ORIGINS = new Set([
  "https://corazonstudioai.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
]);

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

function corsHeaders(request) {
  const origin = request.headers.get("origin") || "";
  return {
    "access-control-allow-origin": ALLOWED_ORIGINS.has(origin)
      ? origin
      : "https://corazonstudioai.github.io",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "Content-Type",
    "access-control-max-age": "86400",
    vary: "Origin",
  };
}

function json(request, data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...JSON_HEADERS, ...corsHeaders(request) },
  });
}

function cleanText(value, max = 2048) {
  return String(value || "").trim().slice(0, max);
}

function isUnsafe(text) {
  const blocked = [
    /sexual\s+(?:con|de)\s+menores/i,
    /pornograf/i,
    /desnud[oa]s?\s+(?:infantil|menor|niñ)/i,
    /abuso\s+sexual/i,
    /cómo\s+(?:matar|fabricar\s+una\s+bomba)/i,
  ];
  return blocked.some((rule) => rule.test(text));
}

function errorMessage(error) {
  const message = String(error?.message || error || "Error desconocido");
  if (message.includes("3036") || message.includes("allocation")) {
    return "Se alcanzó el límite gratuito diario. Intenta nuevamente mañana.";
  }
  if (message.includes("3040") || message.includes("capacity")) {
    return "El servicio gratuito está ocupado. Intenta nuevamente en unos minutos.";
  }
  return "No pudimos completar la solicitud en este momento.";
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function audioDataUri(result) {
  if (typeof result === "string") {
    return result.startsWith("data:") ? result : `data:audio/mpeg;base64,${result}`;
  }
  if (result?.audio) {
    return `data:audio/mpeg;base64,${result.audio}`;
  }
  let buffer;
  if (result instanceof ArrayBuffer) buffer = result;
  else if (ArrayBuffer.isView(result)) {
    buffer = result.buffer.slice(result.byteOffset, result.byteOffset + result.byteLength);
  } else {
    buffer = await new Response(result).arrayBuffer();
  }
  return `data:audio/mpeg;base64,${bytesToBase64(new Uint8Array(buffer))}`;
}

async function readBody(request) {
  const length = Number(request.headers.get("content-length") || 0);
  if (length > 20_000) throw new Error("Solicitud demasiado grande");
  return request.json();
}

async function handleChat(request, env) {
  const body = await readBody(request);
  const message = cleanText(body.message, 4000);
  if (!message) return json(request, { message: "Escribe una idea para comenzar." }, 400);
  if (isUnsafe(message)) return json(request, { message: "Esta solicitud no cumple las reglas de contenido seguro." }, 400);

  const result = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
    messages: [
      {
        role: "system",
        content:
          "Eres Corazón Studio AI. Responde en el idioma del usuario con calidez, claridad y utilidad. Ayuda a crear contenido positivo, educativo, humano y seguro. No produzcas contenido sexual, explotador, violento ni instrucciones peligrosas.",
      },
      { role: "user", content: message },
    ],
    max_tokens: 700,
    temperature: 0.65,
  });
  return json(request, { status: "ok", reply: result.response || "" });
}

async function handleImage(request, env) {
  const body = await readBody(request);
  const prompt = cleanText(body.prompt, 1800);
  if (!prompt) return json(request, { message: "Describe la imagen que deseas crear." }, 400);
  if (isUnsafe(prompt)) return json(request, { message: "Esta solicitud no cumple las reglas de contenido seguro." }, 400);

  const result = await env.AI.run("@cf/black-forest-labs/flux-1-schnell", {
    prompt: `${prompt}. High quality, coherent composition, no watermark, no extra text unless explicitly requested.`,
    steps: 4,
    seed: Math.floor(Math.random() * 999999999) + 1,
  });
  if (!result?.image) throw new Error("El modelo no devolvió una imagen");
  return json(request, {
    status: "ok",
    image_url: `data:image/jpeg;base64,${result.image}`,
  });
}

async function handleTts(request, env) {
  const body = await readBody(request);
  const text = cleanText(body.text, 1200);
  if (!text) return json(request, { message: "Escribe el texto que deseas escuchar." }, 400);
  if (isUnsafe(text)) return json(request, { message: "Esta solicitud no cumple las reglas de contenido seguro." }, 400);
  const lang = body.language === "en" ? "en" : "es";
  const result = await env.AI.run("@cf/myshell-ai/melotts", { prompt: text, lang });
  return json(request, { status: "ok", audio_url: await audioDataUri(result) });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request) });
    }

    const url = new URL(request.url);
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/health")) {
      return json(request, {
        status: "ok",
        message: "Corazón Studio AI gratuito activo",
        provider: "Cloudflare Workers AI",
        features: ["chat", "image", "tts"],
      });
    }

    if (request.method !== "POST") return json(request, { message: "Ruta no encontrada" }, 404);

    try {
      if (url.pathname === "/chat") return await handleChat(request, env);
      if (url.pathname === "/image") return await handleImage(request, env);
      if (url.pathname === "/tts") return await handleTts(request, env);
      if (url.pathname === "/reels" || url.pathname === "/reels-voice" || url.pathname.startsWith("/video-cine")) {
        return json(request, {
          status: "unavailable",
          message: "El video está temporalmente desactivado en el plan gratuito. Texto, imagen y voz continúan disponibles.",
        }, 503);
      }
      return json(request, { message: "Ruta no encontrada" }, 404);
    } catch (error) {
      console.error(error);
      return json(request, { status: "error", message: errorMessage(error) }, 500);
    }
  },
};
