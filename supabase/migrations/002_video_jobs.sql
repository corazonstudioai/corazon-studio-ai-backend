-- Async video jobs for Supabase Edge Functions
create table if not exists public.video_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  reservation_id uuid not null unique references public.generation_events(id),
  provider text not null default 'fal',
  model text not null,
  provider_request_id text not null,
  status_url text not null,
  response_url text not null,
  status text not null default 'queued'
    check (status in ('queued','in_progress','completed','failed')),
  duration_seconds integer not null check (duration_seconds between 1 and 10),
  video_url text,
  error_code text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz
);

create index if not exists video_jobs_user_created_idx
  on public.video_jobs(user_id, created_at desc);

alter table public.video_jobs enable row level security;

create policy "users read own video jobs" on public.video_jobs
  for select using (auth.uid() = user_id);

revoke all on public.video_jobs from anon, authenticated;
grant select on public.video_jobs to authenticated;
grant all on public.video_jobs to service_role;
