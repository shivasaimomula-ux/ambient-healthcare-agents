// Adapted from NVIDIA ace-controller examples/webrtc_ui/src/config.ts (BSD 2-Clause).
// Replaces the upstream file at build time (see setup_ui.sh).

export const RTC_CONFIG = {};

const { hostname, host, protocol } = window.location;
const secure = protocol === "https:";

// Behind the Caddy proxy (https) the voice controller is served at /voice on the same origin, which the browser
// requires for microphone access. In local development the page is on http://localhost and the controller on :7860.
export const RTC_OFFER_URL = secure ? `wss://${host}/voice/ws` : `ws://${hostname}:7860/ws`;
export const POLL_PROMPT_URL = secure ? `https://${host}/voice/get_prompt` : `http://${hostname}:7860/get_prompt`;

// Prompts are owned by the Herbenzo intake agent; the UI must never send one.
export const DYNAMIC_PROMPT = false;
