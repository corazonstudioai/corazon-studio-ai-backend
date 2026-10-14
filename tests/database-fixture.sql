-- Isolated CI fixtures. These are not user accounts or production credentials.
create role anon;
create role authenticated;
create role service_role bypassrls;
create schema auth;
create schema storage;
create table auth.users(id uuid primary key,email_confirmed_at timestamptz,phone_confirmed_at timestamptz);
create function auth.uid() returns uuid language sql as $$ select null::uuid $$;
create function auth.role() returns text language sql as $$ select current_user::text $$;
create table storage.buckets(id text primary key,name text,public boolean,file_size_limit bigint);
