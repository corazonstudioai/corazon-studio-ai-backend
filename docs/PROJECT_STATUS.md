# Corazón Studio IA: estado verificable — 14 de octubre de 2026

## Infraestructura y permisos

Continuar con Supabase; no migrar a Neon. Neon queda como alternativa futura.
La usuaria aprobó integrar y publicar backend/interfaz y realizar una prueba
limitada con OpenAI/FAL a dos créditos de aplicación por archivo.
No se activaron servicios ni facturación nuevos. No se accedió al correo,
teléfono, códigos ni cuentas personales de la usuaria. Las identidades técnicas
de prueba usaron example.invalid, sin enviar correos, y fueron eliminadas.
Las funciones temporales de prueba también fueron eliminadas.

Reglas permanentes: autorización para servicios nuevos, acciones fuera de
GitHub/Supabase y grandes merges a producción; autonomía para auditorías,
correcciones y pruebas internas. Trabajar por bloques, corregir causas raíz
y reportar solo avance total, terminado/probado y decisiones necesarias.

## Publicación completada

- Backend: PR #10 integrado; migración 004 y funciones Edge desplegadas.
- Web: PR #12 integrado y Pages desplegado. Se conservó el cambio reciente
  de «Dinos» a «Dime» resolviendo su conflicto antes de publicar.
- App: https://corazonstudioai.github.io/corazon-studio-ai-web/
- Correcciones derivadas de aceptación: PR #11, #12 y #13 del backend.
  Las tres fueron integradas, comprobadas y desplegadas.

Brain entrega planes estructurados de texto, imagen, voz o video con voz;
valida capacidades y rechaza planes incompletos. El plan cuesta un crédito y
la creación posterior de archivo otro. Cache, identificadores persistentes,
recuperación, Storage privado y reservas/liquidaciones atómicas están integrados.

## Aceptación integral final APROBADA

La ejecución completa del backend final publicado pasó:
https://github.com/corazonstudioai/corazon-studio-ai-backend/actions/runs/38609770823

- Planificación real con OpenAI y recuperación del mismo plan desde cache.
- Una reserva de archivo, replay con el mismo job, generación FAL y voz OpenAI PCM.
- Composición con duración visual completa y voz que termina naturalmente.
- MP4 H.264/AAC descargado y decodificado: 5,031250 segundos de video,
  2,603 segundos de audio. Movimiento continuo en los nueve intervalos
  analizados; mínimo medido 5,00889.
- Ledger real del usuario técnico: exactamente dos eventos, ambos succeeded,
  con un crédito cada uno. billing.json se guardó antes de verificar el archivo.
  Los tres créditos inicialmente utilizados eran el límite artificial de la
  cuenta técnica; no eran consumo de la usuaria.
- Eliminación del usuario técnico y del helper: ambos pasos aprobados.
- No hay nuevas generaciones ni pruebas pagadas automáticas pendientes.
  El workflow de pago queda exclusivamente manual.

Esta aceptación cierra el bloque autorizado de publicación y prueba limitada.
No convierte todos los modos o el producto completo en una aceptación al 100%.

## Causas raíz resueltas durante aceptación

1. El entorno de prueba usaba un token distinto del entorno de despliegue:
   se reutilizó supabase-production, sin exponer credenciales.
2. La cuota gratuita tiene cinco créditos fijos: el usuario técnico se limita
   a dos disponibles mediante tres ya utilizados, sin alterar el esquema.
3. Cinco segundos llegan como 5/60 minutos, pero la base guarda cuatro
   decimales. Se normalizan antes de comparar replay/cuotas; la regresión
   PostgreSQL exige la misma reserva y un único cargo.
4. JSON mode no garantiza el contrato de Brain: OpenAI recibe ahora JSON Schema
   estricto, conservando las validaciones semánticas y devolución por fallo.
5. merge-audio-video recortó el archivo real a 2,35 segundos al terminar la voz.
   compose usa una pista visual explícita y una pista de audio con duración
   medida por muestras PCM 24 kHz. La voz se envuelve en WAV; no se alarga,
   repite ni corta para rellenar el video.
6. El verificador confundía ausencia de muestras después de la voz con fallo:
   mide ese tramo con silencio añadido únicamente durante el análisis.
   No se modificó el contenido del archivo para hacer pasar ese control.

## Evidencia y límites

- Backend: Deno/tipos, integración de handlers con proveedores simulados y
  PostgreSQL 17 aislado aprobados para el último código de producción.
  Incluye concurrencia, permisos, primer uso, hash, cache, devolución,
  duración PCM, respuestas de compose y replay de unidades fraccionarias.
- Web: contratos y navegador móvil aislado aprobados; recuperación, cancelación
  de autenticación ficticia y descarga idéntica byte por byte.
  Pages publicado se comprobó sin iniciar sesión personal.
- Prueba real integral:
  https://github.com/corazonstudioai/corazon-studio-ai-backend/actions/runs/38605430339
  Brain, replay, video y TTS ejecutados; produjo un MP4 real con voz. Esa
  ejecución no pasó su aceptación de duración y se corrigió la causa.
  No etiquetarla como una ejecución integral aprobada ni afirmar que llegó
  al chequeo final del ledger, situado después de la comprobación de duración.
- Composición real corregida:
  https://github.com/corazonstudioai/corazon-studio-ai-backend/actions/runs/38607895496
  Reutilizó la escena y voz existentes, sin nuevas llamadas de generación de
  video ni síntesis de voz. Se ajustó el tiempo visual del fragmento ya
  descargado para comprobar una línea visual de cinco segundos.
- Aceptación del mismo archivo, sin llamadas adicionales a proveedores:
  https://github.com/corazonstudioai/corazon-studio-ai-backend/actions/runs/38608683478
  APROBADA: MP4 de 5,000 segundos, H.264/AAC, movimiento continuo medible,
  voz audible que termina naturalmente y descarga/decodificación completas.
  La validación local confirmó movimiento mínimo 4,10118 y cola de audio
  silenciosa de -91 dB. Los primeros indicadores fallidos del verificador
  no implicaban voz repetida.
- Hubo intentos previos de prueba, incluido un video iniciado antes de
  corregir replay. No afirmar que el gasto total de proveedores fue una sola
  llamada ni confundir créditos de la app con el importe facturado por IA.
  No se consultaron facturas ni se cambió el saldo de la usuaria.

La prueba de pago queda manual, con helper protegido por un token aleatorio y
limpieza de identidades/funciones temporales. No se ejecuta en pushes normales.
La comprobación del ledger se guarda antes de validar duración/media para
conservar evidencia incluso si una futura prueba de archivo falla.

La composición y el despliegue están aceptados. La ejecución integral del backend final, desde planificación hasta ledger, ya
pasó. La aceptación de todos los modos con proveedores reales, dictado, música,
subtítulos, traducción, eliminación de fondo y formatos exactos sigue pendiente. No declarar Brain completo al 100%
ni avanzar a otro bloque atribuyéndole una aceptación que todavía no tiene.

## Porcentaje del plan total

No hay una matriz global ponderada y ejecutada para todo el alcance original.
El porcentaje total sigue sin ser verificable. No convertir conteos de archivos,
commits o pruebas en un porcentaje de producto terminado.
