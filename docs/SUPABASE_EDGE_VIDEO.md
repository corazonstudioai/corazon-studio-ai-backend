# Supabase Edge video

Render is not used by these functions.

- `video-start` authenticates the user, reserves credits atomically, and submits the FAL queue job.
- `video-status` polls FAL and finalizes or refunds the reservation.
- `002_video_jobs.sql` stores durable job state so browser refreshes do not lose progress.

Required Supabase secret: `FAL_KEY`. Never commit it.

Deploy only after migrations 001 and 002 are applied. Both functions require a verified Supabase JWT.

The `Deploy Supabase Edge` GitHub Actions workflow tests the functions, applies the
idempotent migrations through the scoped Management API, and deploys both functions.
It reads `SUPABASE_ACCESS_TOKEN` only from GitHub Actions secrets. The token is never
stored in the repository or printed by the workflow.
