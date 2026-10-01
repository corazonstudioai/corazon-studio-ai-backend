# Supabase Edge backend

Cloudflare Workers and Wrangler are not used by this backend.

- `health` reports service availability without authentication.
- `chat` uses the configured OpenAI text model.
- `image` uses the configured OpenAI image model.
- `tts` uses the configured OpenAI text-to-speech model.

- `video-start` authenticates the user, reserves credits atomically, and submits the FAL queue job.
- `video-status` polls FAL and finalizes or refunds the reservation.
- `002_video_jobs.sql` stores durable job state so browser refreshes do not lose progress.

Required Supabase secrets: `OPENAI_API_KEY` and `FAL_KEY`. Never commit them.
Optional model settings: `OPENAI_MODEL`, `OPENAI_IMAGE_MODEL`, `OPENAI_TTS_MODEL`,
and `VIDEO_MODEL_CINE`.

Deploy only after migrations 001 and 002 are applied. Both functions require a verified Supabase JWT.

The `Deploy Supabase Edge` GitHub Actions workflow tests the functions, applies the
idempotent migrations through the scoped Management API, and deploys all Edge Functions.
It reads `SUPABASE_ACCESS_TOKEN` only from GitHub Actions secrets. The token is never
stored in the repository or printed by the workflow.
