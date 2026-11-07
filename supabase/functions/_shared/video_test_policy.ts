export const WAN_TURBO_MODEL = "fal-ai/wan/v2.2-a14b/text-to-video/turbo";
export const OPENAI_VOICE_MODEL = "gpt-4o-mini-tts";
export const VIDEO_COMPOSE_MODEL = "fal-ai/ffmpeg-api/compose";

export const falSingleAttemptHeaders = {
  "X-Fal-No-Retry": "1",
  "x-app-fal-disable-fallback": "true",
};

export function lockedVideoModelError(videoModel: string, voiceModel: string) {
  if (videoModel !== WAN_TURBO_MODEL) return "video_test_model_mismatch";
  if (voiceModel !== OPENAI_VOICE_MODEL) return "video_test_voice_mismatch";
  return null;
}

// Keep the provider's raw metering separate from a confirmed monetary charge.
// These inference endpoints do not document a per-request monetary total.
export function recordVideoBillingEvidence(
  response: Response,
  provider: "fal" | "openai",
  model: string,
  jobId: string,
  falRequestId?: string,
) {
  const requestId = provider === "fal"
    ? response.headers.get("x-fal-request-id") || falRequestId || null
    : response.headers.get("x-request-id");
  console.info(JSON.stringify({
    event: "locked_video_test_billing",
    job_id: jobId,
    provider,
    model,
    request_id: requestId,
    http_status: response.status,
    billable_units: provider === "fal"
      ? response.headers.get("x-fal-billable-units")
      : null,
    actual_cost_usd: null,
    billing_status: "pending_provider_report",
  }));
}
