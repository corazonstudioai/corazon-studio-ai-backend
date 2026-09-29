import unittest

import httpx

from corazon.billing import (
    BillingUnavailable,
    LimitReached,
    POLICIES,
    SupabaseBilling,
    VerifiedIdentity,
    new_idempotency_key,
)


class BillingTests(unittest.IsolatedAsyncioTestCase):
    def test_video_units_are_measured_in_minutes(self):
        self.assertEqual(POLICIES["/video-cine"].units({"duration": 30}), 0.5)

    def test_idempotency_key_is_validated(self):
        value = new_idempotency_key(None)
        self.assertEqual(new_idempotency_key(value), value)

    async def test_unconfigured_billing_fails_closed(self):
        service = SupabaseBilling()
        with self.assertRaises(BillingUnavailable):
            await service.authenticate("Bearer example")

    async def test_verified_email_identity_is_accepted(self):
        def handler(request):
            self.assertNotIn("service-role", request.headers.get("authorization", ""))
            return httpx.Response(200, json={
                "id": "64bc07da-79dd-4a30-9609-38f30cedefc4",
                "email": "verified@example.test",
                "email_confirmed_at": "2026-01-01T00:00:00Z",
            })
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = SupabaseBilling("https://db.test", "anon", "service-role", client)
            identity = await service.authenticate("Bearer user-token")
        self.assertEqual(identity.kind, "email")

    async def test_limit_response_stops_generation(self):
        def handler(request):
            return httpx.Response(200, json={
                "allowed": False,
                "reason": "credits_exhausted",
            })
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = SupabaseBilling("https://db.test", "anon", "service-role", client)
            with self.assertRaises(LimitReached):
                await service.reserve(
                    VerifiedIdentity("64bc07da-79dd-4a30-9609-38f30cedefc4", "email"),
                    POLICIES["/image"],
                    1,
                    "f39855f6-7f9e-477c-b921-a904849d40ca",
                )


if __name__ == "__main__":
    unittest.main()
