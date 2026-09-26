-- Corazón Studio IA: persistent credits, resource limits and cost ledger
-- Run once in a Supabase project. All mutations occur through atomic RPC functions.
create extension if not exists pgcrypto;

create table if not exists public.billing_accounts (
  user_id uuid primary key references auth.users(id) on delete cascade,
  identity_kind text not null check (identity_kind in ('email','phone')),
  plan_code text not null default 'free',
  free_credits_total integer not null default 5 check (free_credits_total = 5),
  free_credits_used integer not null default 0 check (free_credits_used between 0 and 5),
  purchased_credits integer not null default 0 check (purchased_credits >= 0),
  subscription_credits integer not null default 0 check (subscription_credits >= 0),
  subscription_period_start timestamptz,
  subscription_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.plan_resource_limits (
  plan_code text not null,
  resource text not null,
  limit_units numeric(12,4) not null check (limit_units >= 0),
  primary key (plan_code, resource)
);

insert into public.plan_resource_limits(plan_code, resource, limit_units) values
  ('free','text_requests',5),
  ('free','images',2),
  ('free','video_minutes',0.1667),
  ('free','voice_minutes',1),
  ('free','songs',1)
on conflict (plan_code, resource) do update set limit_units = excluded.limit_units;

create table if not exists public.usage_counters (
  user_id uuid not null references public.billing_accounts(user_id) on delete cascade,
  resource text not null,
  period_key text not null,
  used_units numeric(12,4) not null default 0 check (used_units >= 0),
  primary key (user_id, resource, period_key)
);

create table if not exists public.generation_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.billing_accounts(user_id) on delete cascade,
  idempotency_key uuid not null,
  resource text not null,
  units numeric(12,4) not null check (units > 0),
  credits_charged integer not null check (credits_charged >= 0),
  charged_from text not null check (charged_from in ('free','subscription','purchased')),
  estimated_cost_usd numeric(12,6) not null default 0 check (estimated_cost_usd >= 0),
  engine text not null,
  status text not null default 'reserved' check (status in ('reserved','succeeded','refunded')),
  created_at timestamptz not null default now(),
  completed_at timestamptz,
  unique(user_id, idempotency_key)
);

create index if not exists generation_events_user_created_idx
  on public.generation_events(user_id, created_at desc);

alter table public.billing_accounts enable row level security;
alter table public.usage_counters enable row level security;
alter table public.generation_events enable row level security;
alter table public.plan_resource_limits enable row level security;

create policy "users read own billing account" on public.billing_accounts
  for select using (auth.uid() = user_id);
create policy "users read own counters" on public.usage_counters
  for select using (auth.uid() = user_id);
create policy "users read own generations" on public.generation_events
  for select using (auth.uid() = user_id);
create policy "authenticated users read plan limits" on public.plan_resource_limits
  for select using (auth.role() = 'authenticated');

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
  if p_identity_kind not in ('email','phone') or p_units <= 0 or p_credit_cost < 0 then
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

  select * into existing from public.generation_events
    where user_id = p_user_id and idempotency_key = p_idempotency_key;
  if found then
    select greatest(
      (free_credits_total - free_credits_used) + subscription_credits + purchased_credits, 0
    ) into remaining from public.billing_accounts where user_id = p_user_id;
    return query select true, 'idempotent_replay', existing.id, existing.credits_charged, remaining;
    return;
  end if;

  select * into account from public.billing_accounts
    where user_id = p_user_id for update;

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
    user_id,idempotency_key,resource,units,credits_charged,charged_from,estimated_cost_usd,engine
  ) values (
    p_user_id,p_idempotency_key,p_resource,p_units,p_credit_cost,source,p_estimated_cost_usd,p_engine
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

  select period_key into period from public.usage_counters
    where user_id=event.user_id and resource=event.resource
    order by period_key desc limit 1;
  update public.usage_counters set used_units=greatest(used_units-event.units,0)
    where user_id=event.user_id and resource=event.resource and period_key=period;
  update public.generation_events set status='refunded', completed_at=now() where id=p_reservation_id;
end;
$$;

revoke all on function public.reserve_generation(uuid,text,text,numeric,integer,numeric,text,uuid) from public, anon, authenticated;
revoke all on function public.finalize_generation(uuid,boolean) from public, anon, authenticated;
grant execute on function public.reserve_generation(uuid,text,text,numeric,integer,numeric,text,uuid) to service_role;
grant execute on function public.finalize_generation(uuid,boolean) to service_role;
