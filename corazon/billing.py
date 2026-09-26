"""Fail-closed usage metering backed by Supabase.

Provider calls must never run unless reserve_generation succeeds.
No secrets, identity values, prompts, or generated content are logged here.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import httpx


class BillingError(Exception):
    """Base error safe for translation at the API boundary."""


class BillingUnavailable(BillingError):
    pass


class IdentityRequired(BillingError):
    pass


class LimitReached(BillingError):
    pass


@dataclass(frozen=True)
class ResourcePolicy:
    resource: str
    credit_cost: int
    estimated_cost_usd: float
    engine: str
    unit_field: str | None = None
    unit_divisor: float = 1.0
    minimum_units: float = 1.0

    def units(self, payload: Mapping[str, Any]) -> float:
        raw = payload.get(self.unit_field, self.minimum_units) if self.unit_field else self.minimum_units
        try:
            value = float(raw) / self.unit_divisor
        except (TypeError, ValueError):
            value = self.minimum_units
        return round(max(self.minimum_units, value), 4)


POLICIES: dict[str, ResourcePolicy] = {
    "/chat": ResourcePolicy("text_requests", 1, 0.002, "openai_text"),
    "/image": ResourcePolicy("images", 2, 0.040, "openai_image"),
    "/reels": ResourcePolicy("video_minutes", 1, 0.001, "ffmpeg", "duration", 60.0, 0.0167),
    "/tts": ResourcePolicy("voice_minutes", 1, 0.020, "openai_tts"),
    "/reels-voice": ResourcePolicy("video_minutes", 2, 0.025, "ffmpeg_openai_tts", "duration", 60.0, 0.0167),
    "/video-cine": ResourcePolicy("video_minutes", 5, 0.300, "fal_video", "duration", 60.0, 0.0167),
    "/video-cine-voice": ResourcePolicy("video_minutes", 6, 0.330, "fal_video_openai_tts", "duration", 60.0, 0.0167),
}


@dataclass(frozen=True)
class VerifiedIdentity:
    user_id: str
    kind: str


@dataclass(frozen=True)
class Reservation:
    id: str
    resource: str
    credits_charged: int


class SupabaseBilling:
    def __init__(
        self,
        url: str | None = None,
        anon_key: str | None = None,
        service_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = (url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.anon_key = anon_key or os.getenv("SUPABASE_ANON_KEY", "")
        self.service_key = service_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.url and self.anon_key and self.service_key)

    async def _request(self, method: str, path: str, *, headers: dict[str, str], json: Any | None = None) -> Any:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=15)
        try:
            response = await client.request(method, f"{self.url}{path}", headers=headers, json=json)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BillingUnavailable("billing service unavailable") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def authenticate(self, authorization: str | None) -> VerifiedIdentity:
        if not self.configured:
            raise BillingUnavailable("billing is not configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise IdentityRequired("verified identity required")
        token = authorization[7:].strip()
        if not token:
            raise IdentityRequired("verified identity required")
        data = await self._request(
            "GET",
            "/auth/v1/user",
            headers={"apikey": self.anon_key, "Authorization": f"Bearer {token}"},
        )
        user_id = str(data.get("id", ""))
        email_verified = bool(data.get("email") and data.get("email_confirmed_at"))
        phone_verified = bool(data.get("phone") and data.get("phone_confirmed_at"))
        if not user_id or not (email_verified or phone_verified):
            raise IdentityRequired("verified identity required")
        return VerifiedIdentity(user_id=user_id, kind="phone" if phone_verified else "email")

    async def reserve(
        self,
        identity: VerifiedIdentity,
        policy: ResourcePolicy,
        units: float,
        idempotency_key: str,
    ) -> Reservation:
        data = await self._request(
            "POST",
            "/rest/v1/rpc/reserve_generation",
            headers={
                "apikey": self.service_key,
                "Authorization": f"Bearer {self.service_key}",
                "Content-Type": "application/json",
            },
            json={
                "p_user_id": identity.user_id,
                "p_identity_kind": identity.kind,
                "p_resource": policy.resource,
                "p_units": units,
                "p_credit_cost": policy.credit_cost,
                "p_estimated_cost_usd": policy.estimated_cost_usd,
                "p_engine": policy.engine,
                "p_idempotency_key": idempotency_key,
            },
        )
        row = data[0] if isinstance(data, list) and data else data
        if not isinstance(row, dict) or not row.get("allowed"):
            raise LimitReached(str((row or {}).get("reason", "limit reached")))
        return Reservation(
            id=str(row["reservation_id"]),
            resource=policy.resource,
            credits_charged=int(row.get("credits_charged", policy.credit_cost)),
        )

    async def finalize(self, reservation_id: str, success: bool) -> None:
        await self._request(
            "POST",
            "/rest/v1/rpc/finalize_generation",
            headers={
                "apikey": self.service_key,
                "Authorization": f"Bearer {self.service_key}",
                "Content-Type": "application/json",
            },
            json={"p_reservation_id": reservation_id, "p_success": success},
        )


def new_idempotency_key(value: str | None) -> str:
    if value:
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise IdentityRequired("invalid request identifier") from exc
    return str(uuid.uuid4())
