-- Add the narration and audio-merge stages to asynchronous video jobs.
alter table public.video_jobs
  add column if not exists narration_text text not null default '',
  add column if not exists voice text not null default 'nova',
  add column if not exists raw_video_url text,
  add column if not exists merge_request_id text,
  add column if not exists merge_status_url text,
  add column if not exists merge_response_url text;

alter table public.video_jobs drop constraint if exists video_jobs_voice_check;
alter table public.video_jobs
  add constraint video_jobs_voice_check check (voice in ('nova', 'onyx'));
