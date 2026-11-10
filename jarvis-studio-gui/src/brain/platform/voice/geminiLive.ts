/* Gemini Live listener — ANDROID FORK.
 *
 * Streams microphone audio over Gemini's Live (BidiGenerateContent) WebSocket and
 * returns the transcript of what the user said. Used only for the listening half of
 * a turn. The brain still answers and JARVIS still speaks (webSpeechTTS), so every
 * tool call keeps going through the one dispatcher (CLAUDE.md rule 1).
 *
 * Live's own audio replies are received and ignored: they are never played, so the
 * brain's answer is the only one the user hears.
 */

/** Live model id. Verify against Google's model list before relying on it. */
export const GEMINI_LIVE_MODEL = "gemini-live-2.5-flash-native-audio";

const LIVE_WS_URL =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

const INPUT_RATE = 16000;
/** Quiet this long after the last transcript text → the utterance is final. */
const FINAL_QUIET_MS = 1500;

export interface GeminiLiveOptions {
  apiKey: string;
  model?: string;
  /** Give up if no speech is transcribed within this long. */
  noSpeechMs?: number;
  onFinal: (text: string) => void;
  onNoSpeech?: () => void;
  onError?: (msg: string) => void;
}

function float32ToPcm16Base64(input: Float32Array): string {
  const bytes = new Uint8Array(input.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export class GeminiLiveListener {
  private ws: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private proc: ScriptProcessorNode | null = null;
  private ready = false;
  private transcript = "";
  private quietTimer: ReturnType<typeof setTimeout> | undefined;
  private noSpeechTimer: ReturnType<typeof setTimeout> | undefined;
  private done = false;

  constructor(private opts: GeminiLiveOptions) {}

  get active(): boolean {
    return !this.done && this.ws !== null;
  }

  async start(): Promise<void> {
    const key = (this.opts.apiKey || "").trim();
    if (!key) throw new Error("Gemini Live needs a Gemini API key. Add one in Settings.");
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ws = new WebSocket(`${LIVE_WS_URL}?key=${encodeURIComponent(key)}`);
    this.ws = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          setup: {
            model: `models/${this.opts.model || GEMINI_LIVE_MODEL}`,
            generationConfig: { responseModalities: ["AUDIO"] },
            inputAudioTranscription: {},
          },
        }),
      );
    };
    ws.onmessage = (ev) => void this.onMessage(ev.data);
    ws.onerror = () => this.fail("Gemini Live connection failed.");
    ws.onclose = () => {
      if (!this.done) this.fail("Gemini Live connection closed.");
    };

    this.noSpeechTimer = setTimeout(() => {
      if (!this.transcript) this.finish(false);
    }, this.opts.noSpeechMs ?? 6000);
  }

  private async onMessage(data: unknown): Promise<void> {
    const text = typeof data === "string" ? data : await (data as Blob).text();
    let msg: any;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    if (msg.setupComplete && !this.ready) {
      this.ready = true;
      await this.startMic();
      return;
    }
    if (msg.error) {
      this.fail(msg.error.message || "Gemini Live error.");
      return;
    }
    const piece: string | undefined = msg.serverContent?.inputTranscription?.text;
    if (piece) {
      this.transcript += piece;
      clearTimeout(this.noSpeechTimer);
      clearTimeout(this.quietTimer);
      this.quietTimer = setTimeout(() => this.finish(true), FINAL_QUIET_MS);
    }
  }

  private async startMic(): Promise<void> {
    if (!this.stream) return;
    const ctx = new AudioContext({ sampleRate: INPUT_RATE });
    const source = ctx.createMediaStreamSource(this.stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.done) return;
      const data = float32ToPcm16Base64(e.inputBuffer.getChannelData(0));
      this.ws.send(
        JSON.stringify({
          realtimeInput: { audio: { mimeType: `audio/pcm;rate=${INPUT_RATE}`, data } },
        }),
      );
    };
    source.connect(proc);
    proc.connect(ctx.destination);
    this.ctx = ctx;
    this.proc = proc;
  }

  private finish(hadSpeech: boolean): void {
    if (this.done) return;
    const text = this.transcript.trim();
    this.close();
    if (hadSpeech && text) this.opts.onFinal(text);
    else this.opts.onNoSpeech?.();
  }

  private fail(msg: string): void {
    if (this.done) return;
    this.close();
    this.opts.onError?.(msg);
  }

  /** Finish now with whatever has been transcribed (the user tapped to send). */
  flush(): void {
    this.finish(true);
  }

  /** Stop without delivering anything (the user cancelled, or the mic was handed over). */
  cancel(): void {
    if (this.done) return;
    this.close();
  }

  private close(): void {
    this.done = true;
    clearTimeout(this.quietTimer);
    clearTimeout(this.noSpeechTimer);
    try { this.proc?.disconnect(); } catch { /* ignore */ }
    if (this.ctx) void this.ctx.close().catch(() => {});
    this.stream?.getTracks().forEach((t) => t.stop());
    try { this.ws?.close(); } catch { /* ignore */ }
    this.proc = null;
    this.ctx = null;
    this.stream = null;
    this.ws = null;
  }
}
