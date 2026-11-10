import { afterEach, describe, expect, it, vi } from "vitest";
import { isNoiseTranscript, transcribe } from "./stt";

describe("isNoiseTranscript", () => {
  it("screens an echo of the prompt", () => {
    expect(isNoiseTranscript("A voice command for a phone assistant.")).toBe(true);
    expect(isNoiseTranscript("open YouTube")).toBe(false);
  });
});

afterEach(() => vi.unstubAllGlobals());

it("transcribes a clip with Gemini", async () => {
  const urls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      urls.push(url);
      return new Response(
        JSON.stringify({ candidates: [{ content: { parts: [{ text: "open youtube" }] } }] }),
      );
    }),
  );
  expect(await transcribe(new Blob(["x"], { type: "audio/wav" }), { geminiKey: "AIza-test" })).toBe(
    "open youtube",
  );
  expect(urls[0]).toContain("generativelanguage.googleapis.com");
});

it("refuses to transcribe without a Gemini key", async () => {
  await expect(transcribe(new Blob(["x"]), {})).rejects.toThrow(/Gemini API key/);
});
