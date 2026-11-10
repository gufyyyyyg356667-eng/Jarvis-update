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

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] ?? "");
    r.onerror = () => reject(new Error("Could not read the recording."));
    r.readAsDataURL(blob);
  });
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
  const json: any = await res.json();
  const parts: any[] = json?.candidates?.[0]?.content?.parts ?? [];
  return parts.map((p) => p.text ?? "").join("").trim();
}
