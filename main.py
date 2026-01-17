<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Corazón Studio AI</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:#0b1020;color:#e9eeff}
    .wrap{max-width:980px;margin:0 auto;padding:22px}
    .title{font-size:30px;font-weight:900;margin:10px 0 6px}
    .sub{opacity:.85;margin:0 0 16px}
    .grid{display:grid;grid-template-columns:1fr;gap:14px}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:16px}
    textarea,input,select{width:100%;border-radius:12px;border:1px solid rgba(255,255,255,.16);background:rgba(0,0,0,.25);color:#fff;padding:12px;outline:none}
    button{width:100%;padding:12px 14px;border-radius:12px;border:0;background:#7c3aed;color:white;font-weight:800;cursor:pointer}
    button.secondary{background:rgba(124,58,237,.20);border:1px solid rgba(124,58,237,.40)}
    button:disabled{opacity:.6;cursor:not-allowed}
    .row{display:flex;gap:10px}
    .row>*{flex:1}
    .out{white-space:pre-wrap;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px;min-height:68px}
    .small{font-size:12px;opacity:.85;margin-top:8px}
    img,video{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,.12);margin-top:10px}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);font-size:12px;margin-bottom:10px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="title">Corazón Studio AI ✨</div>
  <p class="sub">Chat, imágenes, reels y video cine realista (texto → video).</p>

  <div class="grid">
    <!-- CHAT -->
    <div class="card">
      <h3>💬 Chat</h3>
      <textarea id="chatText" rows="3" placeholder="Escribe tu mensaje..."></textarea>
      <div style="height:10px"></div>
      <button id="chatBtn">Enviar</button>
      <div style="height:10px"></div>
      <div class="out" id="chatOut"></div>
      <div class="small">Tip: si no responde, revisa que el backend esté activo.</div>
    </div>

    <!-- IMAGEN -->
    <div class="card">
      <h3>🖼️ Imagen (OpenAI)</h3>
      <textarea id="imgText" rows="3" placeholder="Describe la imagen..."></textarea>
      <div style="height:10px"></div>
      <button id="imgBtn">Generar</button>
      <div id="imgOut" class="small"></div>
      <img id="imgPreview" style="display:none" />
      <div class="small">Si no aparece la imagen, te muestro el error exacto abajo.</div>
    </div>

    <!-- REELS -->
    <div class="card">
      <h3>🎞️ Reels MP4 (local)</h3>
      <span class="pill">Opción B: Reels con voz 🎙️</span>

      <textarea id="reelsText" rows="3" placeholder="Texto para el reels..."></textarea>
      <div style="height:10px"></div>

      <div class="row">
        <input id="reelsDur" type="number" value="6" min="1" />
        <select id="reelsFmt">
          <option value="9:16">9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>

      <div style="height:10px"></div>

      <div class="row">
        <select id="voiceSel">
          <option value="alloy">Voz: alloy</option>
          <option value="aria">Voz: aria</option>
          <option value="verse">Voz: verse</option>
        </select>
        <select id="musicLvl">
          <option value="0.10">Música: suave</option>
          <option value="0.15" selected>Música: media</option>
          <option value="0.22">Música: alta</option>
        </select>
      </div>

      <div style="height:10px"></div>

      <button class="secondary" id="reelsBtn">Crear Reels (sin voz) (descargar MP4)</button>
      <div style="height:10px"></div>
      <button id="reelsVoiceBtn">Crear Reels con voz 🎙️ (descargar MP4)</button>

      <div id="reelsOut" class="small"></div>
    </div>

    <!-- VIDEO CINE -->
    <div class="card">
      <h3>🎬 Video cine realista (texto → video)</h3>
      <textarea id="vidText" rows="3" placeholder="Ej: Una niña sonriente caminando en un parque al atardecer, estilo cinematográfico, luz cálida, cámara suave, realista"></textarea>
      <div style="height:10px"></div>

      <div class="row">
        <select id="vidDur">
          <option value="3">3s</option>
          <option value="5" selected>5s</option>
          <option value="6">6s</option>
        </select>
        <select id="vidFmt">
          <option value="9:16" selected>9:16</option>
          <option value="16:9">16:9</option>
        </select>
      </div>

      <div style="height:10px"></div>
      <button id="vidBtn">Generar Video</button>
      <div id="vidOut" class="small"></div>
      <video id="vidPreview" controls playsinline style="display:none"></video>
    </div>
  </div>
</div>

<script>
  const apiBase = ""; // mismo dominio del backend

  function setLoading(btn, on){
    btn.disabled = on;
    btn.textContent = on ? "Procesando..." : btn.dataset.label;
  }

  // CHAT
  const chatBtn = document.getElementById("chatBtn");
  chatBtn.dataset.label = "Enviar";
  chatBtn.onclick = async () => {
    setLoading(chatBtn,true);
    const message = document.getElementById("chatText").value.trim();
    const out = document.getElementById("chatOut");
    out.textContent = "";
    try{
      const r = await fetch(apiBase + "/chat", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({message})
      });
      const data = await r.json();
      out.textContent = data.reply ?? JSON.stringify(data,null,2);
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(chatBtn,false);
  };

  // IMAGEN
  const imgBtn = document.getElementById("imgBtn");
  imgBtn.dataset.label = "Generar";
  imgBtn.onclick = async () => {
    setLoading(imgBtn,true);
    const prompt = document.getElementById("imgText").value.trim();
    const info = document.getElementById("imgOut");
    const img = document.getElementById("imgPreview");
    info.textContent = "";
    img.style.display="none";
    try{
      const r = await fetch(apiBase + "/image", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({prompt})
      });
      const data = await r.json();
      const url = data?.data?.[0]?.url;
      if(url){
        img.src = url;
        img.style.display="block";
        info.textContent = "✅ ¡Imagen lista!";
      }else{
        info.textContent = "⚠️ Respuesta sin url directa: " + JSON.stringify(data,null,2);
      }
    }catch(e){
      info.textContent = "Error: " + e;
    }
    setLoading(imgBtn,false);
  };

  // REELS SIN VOZ
  const reelsBtn = document.getElementById("reelsBtn");
  reelsBtn.dataset.label = "Crear Reels (sin voz) (descargar MP4)";
  reelsBtn.onclick = async () => {
    setLoading(reelsBtn,true);
    const text = document.getElementById("reelsText").value.trim();
    const duration = parseInt(document.getElementById("reelsDur").value || "6",10);
    const format = document.getElementById("reelsFmt").value;
    const out = document.getElementById("reelsOut");
    out.textContent = "";
    try{
      const r = await fetch(apiBase + "/reels", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({text, duration, format})
      });
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "reels.mp4";
      a.click();
      out.textContent = "✅ Listo: descargando reels.mp4 (sin voz)";
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(reelsBtn,false);
  };

  // ✅ REELS CON VOZ
  const reelsVoiceBtn = document.getElementById("reelsVoiceBtn");
  reelsVoiceBtn.dataset.label = "Crear Reels con voz 🎙️ (descargar MP4)";
  reelsVoiceBtn.onclick = async () => {
    setLoading(reelsVoiceBtn,true);
    const text = document.getElementById("reelsText").value.trim();
    const duration = parseInt(document.getElementById("reelsDur").value || "6",10);
    const format = document.getElementById("reelsFmt").value;
    const voice = document.getElementById("voiceSel").value;
    const music_level = parseFloat(document.getElementById("musicLvl").value || "0.15");
    const out = document.getElementById("reelsOut");
    out.textContent = "";
    try{
      const r = await fetch(apiBase + "/reels-voice", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({text, duration, format, voice, music_level})
      });

      // si el backend devuelve error en JSON, intentamos leerlo
      const contentType = r.headers.get("content-type") || "";
      if(contentType.includes("application/json")){
        const data = await r.json();
        out.textContent = "⚠️ Respuesta: " + JSON.stringify(data,null,2);
      } else {
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "reels_con_voz.mp4";
        a.click();
        out.textContent = "✅ Listo: descargando reels_con_voz.mp4 (con voz + música)";
      }
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(reelsVoiceBtn,false);
  };

  // VIDEO CINE
  const vidBtn = document.getElementById("vidBtn");
  vidBtn.dataset.label = "Generar Video";
  vidBtn.onclick = async () => {
    setLoading(vidBtn,true);
    const text = document.getElementById("vidText").value.trim();
    const duration = parseInt(document.getElementById("vidDur").value || "5",10);
    const format = document.getElementById("vidFmt").value;
    const out = document.getElementById("vidOut");
    const vid = document.getElementById("vidPreview");
    out.textContent = "";
    vid.style.display="none";
    try{
      const r = await fetch(apiBase + "/video-cine", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({text, duration, format, cfg: 0.5})
      });
      const data = await r.json();
      if(data.video_url){
        vid.src = data.video_url;
        vid.style.display="block";
        out.textContent = "✅ Video listo";
      }else{
        out.textContent = "⚠️ Respuesta: " + JSON.stringify(data,null,2);
      }
    }catch(e){
      out.textContent = "Error: " + e;
    }
    setLoading(vidBtn,false);
  };
</script>
</body>
</html>
