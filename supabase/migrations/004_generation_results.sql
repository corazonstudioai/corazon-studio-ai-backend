-- Durable results and correct serialized reservations for Brain dependencies.
alter table public.generation_events add column if not exists period_key text;
alter table public.generation_events add column if not exists request_hash text;
update public.generation_events e set period_key = case when a.plan_code = 'free' then 'lifetime' else to_char(e.created_at, 'YYYY-MM') end
from public.billing_accounts a where e.user_id=a.user_id and e.period_key is null;

alter table public.video_jobs add column if not exists merge_claimed_at timestamptz;

create table if not exists public.generation_results (
  reservation_id uuid primary key references public.generation_events(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  result jsonb not null,
  created_at timestamptz not null default now()
);
alter table public.generation_results enable row level security;
revoke all on public.generation_results from anon, authenticated;
grant all on public.generation_results to service_role;
insert into storage.buckets(id,name,public,file_size_limit) values ('creations','creations',false,12582912)
on conflict(id) do nothing;

create or replace function public.reserve_generation(
  p_user_id uuid,
  p_identity_kind text,
  p_resource text,
  p_units numeric,
  p_credit_cost integer,
  p_estimated_cost_usd numeric,
  p_engine text,
  p_idempotency_key uuid
) returns table (
  allowed boolean,
  reason text,
  reservation_id uuid,
  credits_charged integer,
  credits_remaining integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
  account public.billing_accounts%rowtype;
  existing public.generation_events%rowtype;
  resource_limit numeric;
  current_usage numeric := 0;
  source text;
  remaining integer;
  event_id uuid;
  period text;
begin
  -- Match numeric(12,4) storage precision before validating replay or usage.
  p_units := round(p_units, 4);
  if p_identity_kind is null or p_identity_kind not in ('email','phone') or p_units is null or p_units <= 0 or p_credit_cost is null or p_credit_cost < 0 or p_estimated_cost_usd is null or p_estimated_cost_usd < 0 or p_idempotency_key is null then
    return query select false, 'invalid_request', null::uuid, 0, 0;
    return;
  end if;

  if not exists (
    select 1 from auth.users u
    where u.id = p_user_id
      and ((p_identity_kind = 'email' and u.email_confirmed_at is not null)
        or (p_identity_kind = 'phone' and u.phone_confirmed_at is not null))
  ) then
    return query select false, 'identity_not_verified', null::uuid, 0, 0;
    return;
  end if;

  insert into public.billing_accounts(user_id, identity_kind)
  values (p_user_id, p_identity_kind)
  on conflict (user_id) do nothing;

  select * into account from public.billing_accounts
    where user_id = p_user_id for update;

  select * into existing from public.generation_events
    where user_id = p_user_id and idempotency_key = p_idempotency_key;
  if found then
    select greatest(
      (free_credits_total - free_credits_used) + subscription_credits + purchased_credits, 0
    ) into remaining from public.billing_accounts where user_id = p_user_id;
    if existing.status = 'refunded' then
      return query select false, 'generation_already_refunded', existing.id, 0, remaining;
    elsif existing.resource <> p_resource or existing.units <> p_units or existing.credits_charged <> p_credit_cost or existing.engine <> p_engine then
      return query select false, 'idempotency_conflict', existing.id, 0, remaining;
    else
      return query select true, 'idempotent_replay', existing.id, existing.credits_charged, remaining;
    end if;
    return;
  end if;


  period := case when account.plan_code = 'free' then 'lifetime'
                 else to_char(coalesce(account.subscription_period_start, now()), 'YYYY-MM') end;

  select limit_units into resource_limit from public.plan_resource_limits
    where plan_code = account.plan_code and resource = p_resource;
  if resource_limit is null then
    return query select false, 'resource_not_in_plan', null::uuid, 0, 0;
    return;
  end if;

  select coalesce(used_units, 0) into current_usage from public.usage_counters
    where user_id = p_user_id and resource = p_resource and period_key = period;

  current_usage := coalesce(current_usage, 0);

  if current_usage + p_units > resource_limit then
    return query select false, 'resource_limit_reached', null::uuid, 0,
      greatest((account.free_credits_total-account.free_credits_used)+account.subscription_credits+account.purchased_credits,0);
    return;
  end if;

  if account.free_credits_used + p_credit_cost <= account.free_credits_total then
    update public.billing_accounts set free_credits_used = free_credits_used + p_credit_cost, updated_at = now()
      where user_id = p_user_id;
    source := 'free';
  elsif account.subscription_credits >= p_credit_cost then
    update public.billing_accounts set subscription_credits = subscription_credits - p_credit_cost, updated_at = now()
      where user_id = p_user_id;
    source := 'subscription';
  elsif account.purchased_credits >= p_credit_cost then
    update public.billing_accounts set purchased_credits = purchased_credits - p_credit_cost, updated_at = now()
      where user_id = p_user_id;
    source := 'purchased';
  else
    return query select false, 'credits_exhausted', null::uuid, 0, 0;
    return;
  end if;

  insert into public.usage_counters(user_id, resource, period_key, used_units)
    values (p_user_id, p_resource, period, p_units)
    on conflict (user_id, resource, period_key)
    do update set used_units = public.usage_counters.used_units + excluded.used_units;

  insert into public.generation_events(
    user_id,idempotency_key,resource,units,credits_charged,charged_from,estimated_cost_usd,engine,period_key
  ) values (
    p_user_id,p_idempotency_key,p_resource,p_units,p_credit_cost,source,p_estimated_cost_usd,p_engine,period
  ) returning id into event_id;

  select greatest(
    (free_credits_total-free_credits_used)+subscription_credits+purchased_credits,0
  ) into remaining from public.billing_accounts where user_id = p_user_id;

  return query select true, 'reserved', event_id, p_credit_cost, remaining;
end;
$$;

create or replace function public.finalize_generation(
  p_reservation_id uuid,
  p_success boolean
) returns void
language plpgsql
security definer
set search_path = public
as $$
declare
  event public.generation_events%rowtype;
  period text;
begin
  perform 1 from public.billing_accounts where user_id=(select user_id from public.generation_events where id=p_reservation_id) for update;
  select * into event from public.generation_events where id = p_reservation_id for update;
  if not found or event.status <> 'reserved' then return; end if;

  if p_success then
    update public.generation_events set status='succeeded', completed_at=now() where id=p_reservation_id;
    return;
  end if;

  update public.billing_accounts
  set free_credits_used = case when event.charged_from='free' then greatest(free_credits_used-event.credits_charged,0) else free_credits_used end,
      subscription_credits = subscription_credits + case when event.charged_from='subscription' then event.credits_charged else 0 end,
      purchased_credits = purchased_credits + case when event.charged_from='purchased' then event.credits_charged else 0 end,
      updated_at=now()
  where user_id=event.user_id;

  period := event.period_key;
  update public.usage_counters set used_units=greatest(used_units-event.units,0)
    where user_id=event.user_id and resource=event.resource and period_key=period;
  update public.generation_events set status='refunded', completed_at=now() where id=p_reservation_id;
end;
$$;

create or replace function public.reserve_generation_v2(
  p_user_id uuid, p_identity_kind text, p_resource text, p_units numeric,
  p_credit_cost integer, p_estimated_cost_usd numeric, p_engine text,
  p_idempotency_key uuid, p_request_hash text
) returns table(allowed boolean, reason text, reservation_id uuid, credits_charged integer, credits_remaining integer)
language plpgsql security definer set search_path=public as $$
declare receipt record; existing_hash text;
begin
  if p_request_hash is null or p_request_hash !~ '^[0-9a-f]{64}$' then
    return query select false,'invalid_request',null::uuid,0,0; return;
  end if;
  select * into receipt from public.reserve_generation(p_user_id,p_identity_kind,p_resource,p_units,p_credit_cost,p_estimated_cost_usd,p_engine,p_idempotency_key);
  if receipt.allowed then
    select request_hash into existing_hash from public.generation_events where id=receipt.reservation_id for update;
    if existing_hash is not null and existing_hash <> p_request_hash then
      return query select false,'idempotency_conflict',receipt.reservation_id,0,receipt.credits_remaining; return;
    end if;
    update public.generation_events set request_hash=p_request_hash where id=receipt.reservation_id and request_hash is null;
  end if;
  return query select receipt.allowed,receipt.reason,receipt.reservation_id,receipt.credits_charged,receipt.credits_remaining;
end; $$;

create or replace function public.save_generation_result(p_reservation_id uuid,p_result jsonb)
returns void language plpgsql security definer set search_path=public as $$
declare event public.generation_events%rowtype;
begin
  perform 1 from public.billing_accounts where user_id=(select user_id from public.generation_events where id=p_reservation_id) for update;
  select * into event from public.generation_events where id=p_reservation_id for update;
  if not found or event.status='refunded' or p_result is null then raise exception 'invalid_generation_result'; end if;
  insert into public.generation_results(reservation_id,user_id,result)
  values(event.id,event.user_id,p_result) on conflict(reservation_id) do nothing;
  perform public.finalize_generation(event.id,true);
end; $$;
revoke all on function public.reserve_generation_v2(uuid,text,text,numeric,integer,numeric,text,uuid,text) from public,anon,authenticated;
revoke all on function public.save_generation_result(uuid,jsonb) from public,anon,authenticated;
grant execute on function public.reserve_generation_v2(uuid,text,text,numeric,integer,numeric,text,uuid,text) to service_role;
grant execute on function public.save_generation_result(uuid,jsonb) to service_role;

-- A video terminal state and its credit settlement must commit together.
create or replace function public.settle_video_job(p_job_id uuid,p_success boolean,p_video_url text default null,p_error_code text default null)
returns jsonb language plpgsql security definer set search_path=public as $$
declare job public.video_jobs%rowtype;
begin
  perform 1 from public.billing_accounts where user_id=(select user_id from public.video_jobs where id=p_job_id) for update;
  perform 1 from public.generation_events where id=(select reservation_id from public.video_jobs where id=p_job_id) for update;
  select * into job from public.video_jobs where id=p_job_id for update;
  if not found then raise exception 'video_job_not_found'; end if;
  if job.status not in ('completed','failed') then
    if p_success and (p_video_url is null or p_video_url !~ '^https://') then raise exception 'invalid_video_url'; end if;
    update public.video_jobs set status=case when p_success then 'completed' else 'failed' end,
      video_url=case when p_success then p_video_url else null end,
      error_code=case when p_success then null else p_error_code end,
      updated_at=now(),completed_at=now() where id=p_job_id returning * into job;
  end if;
  perform public.finalize_generation(job.reservation_id,job.status='completed');
  return jsonb_build_object('status',job.status,'video_url',job.video_url,'has_audio',job.status='completed','refunded',job.status='failed');
end; $$;
revoke all on function public.settle_video_job(uuid,boolean,text,text) from public,anon,authenticated;
grant execute on function public.settle_video_job(uuid,boolean,text,text) to service_role;
