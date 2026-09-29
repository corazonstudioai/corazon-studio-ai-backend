"""FastAPI dependency that reserves usage before any generation starts."""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Header, HTTPException, Request

from .billing import (
    BillingUnavailable,
    IdentityRequired,
    LimitReached,
    POLICIES,
    SupabaseBilling,
    new_idempotency_key,
)

billing = SupabaseBilling()


async def require_generation_budget(
    request: Request,
    authorization: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AsyncIterator[None]:
    policy = POLICIES.get(request.url.path)
    if policy is None:
        yield
        return

    try:
        payload = await request.json()
        identity = await billing.authenticate(authorization)
        reservation = await billing.reserve(
            identity,
            policy,
            policy.units(payload if isinstance(payload, dict) else {}),
            new_idempotency_key(idempotency_key),
        )
    except IdentityRequired as exc:
        raise HTTPException(
            status_code=401,
            detail="Verifica tu correo o teléfono para usar tus créditos.",
        ) from exc
    except LimitReached as exc:
        raise HTTPException(
            status_code=402,
            detail="Llegaste al límite disponible. Puedes comprar créditos o cambiar de plan.",
        ) from exc
    except BillingUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="No podemos comprobar tu saldo en este momento. No se realizó ningún cargo.",
        ) from exc

    try:
        yield
    except Exception:
        try:
            await billing.finalize(reservation.id, success=False)
        except BillingUnavailable:
            pass
        raise
    else:
        try:
            await billing.finalize(reservation.id, success=True)
        except BillingUnavailable:
            # The generation completed. Reconciliation can recover reserved rows.
            pass
