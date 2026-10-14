\set ON_ERROR_STOP on
insert into auth.users(id,email_confirmed_at) values
('00000000-0000-4000-8000-000000000001',now()),
('00000000-0000-4000-8000-000000000002',now()),
('00000000-0000-4000-8000-000000000003',now());

do $$ declare r record; event_id uuid; used numeric; status text; begin
  -- A NULL usage counter must not bypass the first-ever quota check.
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000001','email','voice_minutes',99,1,0.01,'test','10000000-0000-4000-8000-000000000001',repeat('a',64));
  assert not r.allowed and r.reason='resource_limit_reached','first request bypassed its quota';
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000001','email','text_requests',1,1,0.01,'test','10000000-0000-4000-8000-000000000002',repeat('b',64));
  assert r.allowed and r.reason='reserved' and r.credits_remaining=4;
  event_id := r.reservation_id;
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000001','email','text_requests',1,1,0.01,'test','10000000-0000-4000-8000-000000000002',repeat('b',64));
  assert r.allowed and r.reason='idempotent_replay' and r.reservation_id=event_id and r.credits_remaining=4;
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000001','email','text_requests',1,1,0.01,'test','10000000-0000-4000-8000-000000000002',repeat('c',64));
  assert not r.allowed and r.reason='idempotency_conflict';
  perform public.save_generation_result(event_id,'{"reply":"saved","status":"ok"}'::jsonb);
  select e.status into status from public.generation_events e where id=event_id;
  assert status='succeeded';
  perform public.finalize_generation(event_id,false);
  select free_credits_used into used from public.billing_accounts where user_id='00000000-0000-4000-8000-000000000001';
  assert used=1,'a completed request was refunded';
  assert exists(select 1 from public.generation_results where reservation_id=event_id and result->>'reply'='saved');
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000002','email','text_requests',1,1,0.01,'test','10000000-0000-4000-8000-000000000003',repeat('d',64));
  event_id:=r.reservation_id;
  perform public.finalize_generation(event_id,false);
  perform public.finalize_generation(event_id,false);
  select free_credits_used into used from public.billing_accounts where user_id='00000000-0000-4000-8000-000000000002';
  assert used=0,'refund is not idempotent';
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000002','email','text_requests',1,1,0.01,'test','10000000-0000-4000-8000-000000000003',repeat('d',64));
  assert not r.allowed and r.reason='generation_already_refunded';
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000002','email','text_requests',1,1,0.01,'test','10000000-0000-4000-8000-000000000004',null);
  assert not r.allowed and r.reason='invalid_request';
  assert not has_function_privilege('authenticated','public.reserve_generation_v2(uuid,text,text,numeric,integer,numeric,text,uuid,text)','execute');
  assert not has_table_privilege('authenticated','public.generation_results','select');
  assert exists(select 1 from storage.buckets where id='creations' and not public);
end $$;

do $$ declare r record; event_id uuid; video_id uuid; terminal jsonb; used integer; begin
  select * into r from public.reserve_generation_v2('00000000-0000-4000-8000-000000000002','email','video_minutes',0.0833,1,0.1,'test','10000000-0000-4000-8000-000000000005',repeat('f',64));
  assert r.allowed;
  event_id:=r.reservation_id;
  insert into public.video_jobs(user_id,reservation_id,model,provider_request_id,status_url,response_url,duration_seconds,narration_text)
  values('00000000-0000-4000-8000-000000000002',event_id,'test','test-id','https://queue.fal.run/status','https://queue.fal.run/result',5,'Hola') returning id into video_id;
  select public.settle_video_job(video_id,true,'https://example.invalid/video.mp4',null) into terminal;
  assert terminal->>'status'='completed';
  select public.settle_video_job(video_id,false,null,'late_timeout') into terminal;
  assert terminal->>'status'='completed','a late poll overwrote completion';
  select free_credits_used into used from public.billing_accounts where user_id='00000000-0000-4000-8000-000000000002';
  assert used=1,'a completed video was refunded by a later poll';
end $$;
