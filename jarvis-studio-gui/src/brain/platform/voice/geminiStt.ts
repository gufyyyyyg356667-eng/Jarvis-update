/* Gemini speech-to-text for recorded audio — ANDROID FORK.
 *
 * Sends one recorded clip (the wake command, or a push-to-talk recording) to Gemini
 * and returns the transcript. Used when the voice engine is Gemini (the default);
 * the Groq/Vertex path remains as a fallback in stt.ts.
 */

/** Gemini model used for transcription. Verify against Google's model list. */
/** Models tried in order. A model Google has retired answers 404, and the next one is tried. */
export const GEMINI_STT_MODELS = ["gemini-flash-latest", "gemini-3.1-flash-lite-preview"];
const URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

const INSTRUCTION =
  "Transcribe this audio exactly as spoken. The speaker's main language is Bengali (Bangla). " +
  "Bengali can sound close to Hindi: if the speech could be either, it is Bengali. Write it as " +
  "Hindi (Devanagari) only when the speaker clearly speaks Hindi. The speaker may also use " +
  "English, or mix languages in one sentence. Write Bengali in Bengali script and English in " +
  "Latin letters. A wake phrase such as 'Hey Jarvis' may appear in any script " +
  "(for example হে জার্ভিস or हे जार्विस); write it as spoken. " +
  "Output only the transcript, with no quotes or commentary. If there is no speech, output nothing.";

/** Answers that mean "busy right now, try again" rather than "this model is gone". */
const BUSY = new Set([429, 500, 503, 504]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface GeminiTextResponse {
  candidates?: { content?: { parts?: { text?: string }[] } }[];
}

async function blobToBase64(blob: Blob): Promise<string> {
  // arrayBuffer() works in the phone's WebView and in Node tests alike (FileReader does not run in Node).
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

export async function transcribeWithGemini(blob: Blob, apiKey: string): Promise<string> {
  const key = (apiKey || "").trim();
  if (!key) throw new Error("Gemini transcription needs a Gemini API key.");
  const data = await blobToBase64(blob);
  const body = JSON.stringify({
    contents: [
      {
        parts: [
          { text: INSTRUCTION },
          { inlineData: { mimeType: blob.type || "audio/wav", data } },
        ],
      },
    ],
    generationConfig: { temperature: 0 },
  });
  let lastStatus = 0;
  for (const model of GEMINI_STT_MODELS) {
    // Busy answers (503 overloaded, 429 rate limit) are retried with a short wait.
    for (let attempt = 0; attempt < 3; attempt++) {
      const res = await fetch(`${URL_BASE}/${model}:generateContent`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": key },
        body,
      });
      lastStatus = res.status;
      if (res.status === 404) break; // retired on the API: try the next model
      if (BUSY.has(res.status)) {
        await sleep(800 * (attempt + 1));
        continue;
      }
      if (!res.ok) throw new Error(`Gemini transcription failed (HTTP ${res.status}).`);
      const json = (await res.json()) as GeminiTextResponse;
      const parts = json.candidates?.[0]?.content?.parts ?? [];
      return parts.map((p) => p.text ?? "").join("").trim();
    }
  }
  throw new Error(`Gemini transcription failed (HTTP ${lastStatus}). Gemini is busy or unavailable; try again.`);
}
