// Persist results before reporting success. Replays return the saved result and
// never start a second paid provider request.
export async function replayResult(service: any, reservation: any) {
  if (reservation.reason !== "idempotent_replay") return null;
  const { data, error } = await service.from("generation_results").select("result").eq("reservation_id", reservation.reservation_id).maybeSingle();
  if (error) throw new Error("result_storage_unavailable");
  if (!data) throw new Error("generation_in_progress");
  return hydrateResult(service, data.result);
}

export async function hydrateResult(service: any, result: Record<string, unknown>) {
  if (!result.storage_path) return result;
  const { data, error } = await service.storage.from("creations").createSignedUrl(result.storage_path, 3600);
  if (error || !data?.signedUrl) throw new Error("result_storage_unavailable");
  return { status: "ok", [String(result.media_field)]: data.signedUrl };
}

export async function saveResult(service: any, reservationId: string, result: Record<string, unknown>) {
  const { error } = await service.rpc("save_generation_result", { p_reservation_id: reservationId, p_result: result });
  if (error) throw new Error("result_storage_unavailable");
}

export async function saveMediaResult(service: any, reservationId: string, field: string, bytes: Uint8Array, mime: string) {
  if (!bytes.length || bytes.length > 12 * 1024 * 1024) throw new Error("invalid_media_result");
  const extension = mime === "audio/mpeg" ? "mp3" : "png";
  const path = `${reservationId}/result.${extension}`;
  const { error } = await service.storage.from("creations").upload(path, bytes, { contentType: mime, upsert: false });
  if (error) throw new Error("result_storage_unavailable");
  const stored = { status: "ok", storage_path: path, media_field: field };
  const hydrated = await hydrateResult(service, stored);
  await saveResult(service, reservationId, stored);
  return hydrated;
}
