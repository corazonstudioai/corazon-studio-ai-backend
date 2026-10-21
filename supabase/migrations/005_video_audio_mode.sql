-- Preserve narrated legacy jobs and explicitly mark new silent compositions.
alter table public.video_jobs
  add column if not exists audio_mode text not null default 'narrated';
alter table public.video_jobs drop constraint if exists video_jobs_audio_mode_check;
alter table public.video_jobs add constraint video_jobs_audio_mode_check
  check (audio_mode in ('narrated', 'silent'));
