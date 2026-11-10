/* Gemini speech-to-text for recorded audio — ANDROID FORK.
 *
 * Sends one recorded clip (the wake command, or a push-to-talk recording) to Gemini
 * and returns the transcript. Used when the voice engine is Gemini (the default);
 * the Groq/Vertex path remains as a fallback in stt.ts.
 */

/** Gemini model used for transcription. Verify against Google's model list. */
export const GEMINI_STT_MODEL = "gemini-2.5-flash";
const URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

const INSTRUCTION =
  "Transcribe this audio exactly as spoken, in the language it is spoken in. " +
  "Output only the transcript, with no quotes or commentary. If there is no speech, output nothing.";

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
  const res = await fetch(`${URL_BASE}/${GEMINI_STT_MODEL}:generateContent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({
      contents: [
        {
          parts: [
            { text: INSTRUCTION },
            { inlineData: { mimeType: blob.type || "audio/wav", data } },
          ],
        },
      ],
      generationConfig: { temperature: 0 },
    }),
  });
  if (!res.ok) throw new Error(`Gemini transcription failed (HTTP ${res.status}).`);
  const json = (await res.json()) as GeminiTextResponse;
  const parts = json.candidates?.[0]?.content?.parts ?? [];
  return parts.map((p) => p.text ?? "").join("").trim();
}
