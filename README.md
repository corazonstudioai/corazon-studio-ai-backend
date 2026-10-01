# Corazón Studio AI backend

Backend basado en Supabase Edge Functions:

- `health`: disponibilidad del servicio.
- `chat`: generación de texto.
- `image`: generación de imágenes.
- `tts`: síntesis de voz.
- `video-start` y `video-status`: video asíncrono con seguimiento persistente.

La autenticación, los créditos y los límites se administran en Supabase. Las
claves de los proveedores se guardan exclusivamente como secretos de Supabase.
