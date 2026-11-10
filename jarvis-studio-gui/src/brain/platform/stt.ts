/* Speech-to-text helpers — ANDROID FORK.
 *
 * Voice is Gemini-only. `transcribe` sends a recorded clip to Gemini; the phrase
 * filter below screens the stock phrases speech models invent on silence.
 */

import { transcribeWithGemini } from "./voice/geminiStt";

/** Transcribe a recorded clip with Gemini. */
export async function transcribe(
  blob: Blob,
  keys: { geminiKey?: string; prompt?: string },
): Promise<string> {
  const key = (keys.geminiKey || "").trim();
  if (!key) throw new Error("Voice needs a Gemini API key. Add one in Settings.");
  return transcribeWithGemini(blob, key);
}

/**
 * Speech models reliably "hallucinate" a short stock phrase when handed near-
 * silence or unintelligible noise — most often "Thank you." / "Thanks for watching!" /
 * "you" / "Bye." — because those dominate its training data's trailing frames. After
 * the wake word, an empty command window would otherwise fire one of these as a real
 * command (JARVIS replying to a phantom "thank you"). This screens them out. Kept to
 * whole-utterance matches so it never eats a genuine short command like "call mom".
 */
const HALLUCINATION_PHRASES = new Set([
  "thank you",
  "thank you.",
  "thanks",
  "thanks.",
  "thank you very much",
  "thanks for watching",
  "thanks for watching!",
  "thank you for watching",
  "thank you for watching.",
  "please subscribe",
  "you",
  "bye",
  "bye.",
  "bye bye",
  "goodbye",
  "so",
  "okay",
  "ok",
  "the",
  "i'm sorry",
  "silence",
  "[silence]",
  "[ silence ]",
  "music",
  "[music]",
  "[ music ]",
  "applause",
  "[applause]",
  "you're welcome",
  "see you next time",
]);

// The wake word may already be stripped, so match the prompt's tail.
const PROMPT_ECHO = /^(hey jarvis )?a voice command for a phone assistant$/;

export function isNoiseTranscript(text: string): boolean {
  const norm = (text || "")
    .toLowerCase()
    .replace(/[.!?,;:"'”“]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!norm) return true;
  return HALLUCINATION_PHRASES.has(norm) || PROMPT_ECHO.test(norm);
}

