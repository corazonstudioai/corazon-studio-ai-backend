"""Exercise concurrent real PostgreSQL reservations in the isolated CI database."""
import concurrent.futures
import subprocess

USER = "00000000-0000-4000-8000-000000000003"
KEY = "10000000-0000-4000-8000-000000000099"


def query(sql):
    return subprocess.check_output(["psql", "-X", "-At", "-v", "ON_ERROR_STOP=1", "-c", sql], text=True).strip()


def reserve(_):
    return query(f"select reason from public.reserve_generation_v2('{USER}','email','text_requests',1,1,0.01,'test','{KEY}',repeat('e',64))")


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    replies = list(executor.map(reserve, range(8)))
assert replies.count("reserved") == 1, replies
assert replies.count("idempotent_replay") == 7, replies
assert query(f"select count(*) from public.generation_events where user_id='{USER}'") == "1"
assert query(f"select free_credits_used from public.billing_accounts where user_id='{USER}'") == "1"
assert query(f"select used_units from public.usage_counters where user_id='{USER}' and resource='text_requests'") == "1.0000"
print("8 concurrent submissions: one reservation, one charge, one usage unit.")
