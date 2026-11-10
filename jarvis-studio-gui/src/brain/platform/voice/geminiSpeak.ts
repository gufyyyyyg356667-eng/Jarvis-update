/* Gemini voice output — ANDROID FORK.
 *
 * Speaks JARVIS's replies with Gemini's text-to-speech model, so the voice the user
 * hears comes from Gemini. The text is the brain's answer, so tools and rules are
 * unchanged. If Gemini speech fails (no key, quota, network), it falls back to the
 * device voice (webSpeechTTS), so JARVIS never goes silent.
 */

import { platformFetch } from "../../tools/httpClient";
import { stripMarkdownForSpeech } from "../../formatting";

/** Gemini TTS model id. Verify against Google's model list before relying on it. */
export const GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts";
/** Prebuilt voice name. Pick another from Google's voice list if preferred. */
export const GEMINI_TTS_VOICE = "Kore";

const TTS_RATE = 24000; // Gemini returns 16-bit mono PCM at 24 kHz
const TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models";

let ctx: AudioContext | null = null;
let current: AudioBufferSourceNode | null = null;

function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength >> 1);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
  return out;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** Fetch Gemini's speech for `text` as PCM bytes. */
export async function fetchGeminiSpeech(text: string, apiKey: string): Promise<Uint8Array> {
  const key = (apiKey || "").trim();
  if (!key) throw new Error("Gemini voice needs a Gemini API key.");
  const res = await platformFetch(20000)(`${TTS_URL}/${GEMINI_TTS_MODEL}:generateContent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({
      contents: [{ parts: [{ text }] }],
      generationConfig: {
        responseModalities: ["AUDIO"],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: GEMINI_TTS_VOICE } } },
      },
    }),
  });
  if (!res.ok) throw new Error(`Gemini speech failed (HTTP ${res.status}).`);
  const data: any = await res.json();
  const b64 = data?.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!b64) throw new Error("Gemini speech returned no audio.");
  return base64ToBytes(b64);
}

/** Play PCM bytes and resolve when playback ends (or is stopped). */
export function playPcm(pcm: Uint8Array): Promise<void> {
  return new Promise((resolve) => {
    try {
      if (!ctx) ctx = new AudioContext();
      const samples = pcm16ToFloat32(pcm);
      const buffer = ctx.createBuffer(1, samples.length, TTS_RATE);
      buffer.getChannelData(0).set(samples);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      current = src;
      src.onended = () => {
        if (current === src) current = null;
        resolve();
      };
      src.start();
    } catch {
      resolve();
    }
  });
}

/** Stop any Gemini speech that is playing. */
export function stopGeminiSpeech(): void {
  try {
    current?.stop();
  } catch {
    /* already stopped */
  }
  current = null;
}

/** Stop JARVIS speaking. */
export function stopSpeech(): void {
  stopGeminiSpeech();
}

export interface VoiceConfig {
  geminiKey?: string;
}

let voiceConfig: VoiceConfig = {};

/** Called whenever the saved voice settings change, so any caller can speak. */
export function setVoiceConfig(cfg: VoiceConfig): void {
  voiceConfig = cfg;
}

/** Speak a reply with Gemini. Failures are logged; they never break the turn. */
export async function speakReply(text: string, cfg: VoiceConfig = voiceConfig): Promise<void> {
  const clean = stripMarkdownForSpeech(text || "");
  if (!clean) return;
  const key = (cfg.geminiKey || "").trim();
  if (!key) {
    console.warn("Voice output needs a Gemini API key.");
    return;
  }
  try {
    const pcm = await fetchGeminiSpeech(clean, key);
    await playPcm(pcm);
  } catch (err) {
    console.warn("Gemini voice failed:", err);
  }
}
