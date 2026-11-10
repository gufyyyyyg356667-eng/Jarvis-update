/* Gemini Live voice-to-voice session — ANDROID FORK.
 *
 * One full-duplex Live session: the microphone streams up, and the model's spoken
 * answers stream back and play at once. The user can interrupt (barge-in), and the
 * Live model's function calls are run by the brain's dispatcher through `callTool`,
 * so tools keep the same validation and execution path as text chat.
 */

export type LiveState = "connecting" | "listening" | "thinking" | "speaking" | "idle";

/** The parts of Gemini Live's server messages that this file reads. */
interface LivePart {
  text?: string;
  inlineData?: { data?: string; mimeType?: string };
}
interface LiveFunctionCall {
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
}
interface LiveServerContent {
  interrupted?: boolean;
  turnComplete?: boolean;
  modelTurn?: { parts?: LivePart[] };
  inputTranscription?: { text?: string };
  outputTranscription?: { text?: string };
}
interface LiveMessage {
  setupComplete?: unknown;
  goAway?: unknown;
  error?: { message?: string };
  toolCall?: { functionCalls?: LiveFunctionCall[] };
  serverContent?: LiveServerContent;
}

export interface LiveSessionOptions {
  apiKey: string;
  model: string;
  voiceName: string;
  instruction: string;
  tools: Record<string, unknown>[];
  /** Runs one function call through the brain. Its result is sent back to the model. */
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  onUserText?: (text: string) => void;
  onAssistantText?: (text: string) => void;
  onState?: (state: LiveState) => void;
  onError?: (message: string) => void;
  onClosed?: () => void;
}

/** Live model and voice. Verify both against Google's model and voice lists. */
export const LIVE_MODEL = "gemini-live-2.5-flash-native-audio";
export const LIVE_VOICE = "Kore";

const LIVE_WS =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const MIC_RATE = 16000;
const SPEAKER_RATE = 24000;
const MAX_RECONNECTS = 3;
const MAX_TOOL_TEXT = 4000;

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

function base64ToFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const view = new DataView(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
  const out = new Float32Array(bin.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
  return out;
}

function clip(value: unknown): unknown {
  let text: string;
  try {
    text = JSON.stringify(value ?? null) ?? "null";
  } catch {
    text = String(value);
  }
  if (text.length <= MAX_TOOL_TEXT) return JSON.parse(text);
  return { truncated: true, text: text.slice(0, MAX_TOOL_TEXT) };
}

export class LiveVoiceSession {
  private ws: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private micCtx: AudioContext | null = null;
  private proc: ScriptProcessorNode | null = null;
  private playCtx: AudioContext | null = null;
  private playAt = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private ready = false;
  private stopped = false;
  private muted = false;
  private reconnects = 0;
  private userBuf = "";
  private assistantBuf = "";
  private state: LiveState = "connecting";

  constructor(private opts: LiveSessionOptions) {}

  get active(): boolean {
    return !this.stopped;
  }

  async start(): Promise<void> {
    const key = (this.opts.apiKey || "").trim();
    if (!key) throw new Error("Live voice needs a Gemini API key.");
    // Echo cancellation keeps JARVIS's own voice from being heard back as a command.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    await this.startMic();
    this.connect();
  }

  setMuted(on: boolean): void {
    this.muted = on;
  }

  /** Send typed or wake-command text into the live conversation. */
  sendText(text: string): void {
    const t = text.trim();
    if (!t || !this.isOpen()) return;
    this.ws?.send(
      JSON.stringify({
        clientContent: { turns: [{ role: "user", parts: [{ text: t }] }], turnComplete: true },
      }),
    );
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.stopPlayback();
    try { this.proc?.disconnect(); } catch { /* ignore */ }
    if (this.micCtx) void this.micCtx.close().catch(() => {});
    if (this.playCtx) void this.playCtx.close().catch(() => {});
    this.stream?.getTracks().forEach((t) => t.stop());
    try { this.ws?.close(); } catch { /* ignore */ }
    this.proc = null;
    this.micCtx = null;
    this.playCtx = null;
    this.stream = null;
    this.ws = null;
    this.opts.onClosed?.();
  }

  private isOpen(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  private setState(s: LiveState): void {
    if (this.state === s) return;
    this.state = s;
    this.opts.onState?.(s);
  }

  private connect(): void {
    const ws = new WebSocket(`${LIVE_WS}?key=${encodeURIComponent(this.opts.apiKey.trim())}`);
    this.ws = ws;
    this.ready = false;
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          setup: {
            model: `models/${this.opts.model}`,
            generationConfig: {
              responseModalities: ["AUDIO"],
              speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: this.opts.voiceName } } },
            },
            systemInstruction: { parts: [{ text: this.opts.instruction }] },
            ...(this.opts.tools.length ? { tools: this.opts.tools } : {}),
            inputAudioTranscription: {},
            outputAudioTranscription: {},
          },
        }),
      );
    };
    ws.onmessage = (ev) => void this.onMessage(ev.data);
    ws.onerror = () => {
      /* the close handler decides what to do */
    };
    ws.onclose = () => {
      if (this.stopped) return;
      this.ready = false;
      if (this.reconnects < MAX_RECONNECTS) {
        this.reconnects++;
        this.setState("connecting");
        setTimeout(() => {
          if (!this.stopped) this.connect();
        }, 800 * this.reconnects);
      } else {
        this.opts.onError?.("Live voice connection closed.");
        this.stop();
      }
    };
  }

  private async onMessage(data: unknown): Promise<void> {
    const text = typeof data === "string" ? data : await (data as Blob).text();
    let msg: LiveMessage;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    if (msg.setupComplete) {
      this.ready = true;
      this.reconnects = 0;
      this.setState("listening");
      return;
    }
    if (msg.error) {
      this.opts.onError?.(msg.error.message || "Live voice error.");
      return;
    }
    if (msg.goAway) {
      try { this.ws?.close(); } catch { /* onclose reconnects */ }
      return;
    }
    if (msg.toolCall?.functionCalls) {
      this.setState("thinking");
      void this.runCalls(msg.toolCall.functionCalls);
      return;
    }
    const sc = msg.serverContent;
    if (!sc) return;
    if (sc.interrupted) {
      this.stopPlayback(); // barge-in: the user spoke over JARVIS
      this.setState("listening");
    }
    const parts: LivePart[] = sc.modelTurn?.parts ?? [];
    for (const part of parts) {
      const audio = part.inlineData?.data;
      if (audio && String(part.inlineData?.mimeType || "").startsWith("audio/")) {
        this.setState("speaking");
        this.playChunk(base64ToFloat32(audio));
      }
    }
    if (sc.inputTranscription?.text) this.userBuf += sc.inputTranscription.text;
    if (sc.outputTranscription?.text) this.assistantBuf += sc.outputTranscription.text;
    if (sc.turnComplete || sc.interrupted) {
      if (this.userBuf.trim()) this.opts.onUserText?.(this.userBuf.trim());
      if (this.assistantBuf.trim()) this.opts.onAssistantText?.(this.assistantBuf.trim());
      this.userBuf = "";
      this.assistantBuf = "";
      if (sc.turnComplete) this.setState(this.playingNow() ? "speaking" : "listening");
    }
  }

  private async runCalls(calls: LiveFunctionCall[]): Promise<void> {
    const responses = [];
    for (const call of calls) {
      let response: unknown;
      try {
        response = { result: clip(await this.opts.callTool(String(call.name), call.args ?? {})) };
      } catch (err) {
        response = { error: (err as Error)?.message || String(err) };
      }
      responses.push({ id: call.id, name: call.name, response });
    }
    if (this.isOpen() && !this.stopped) {
      this.ws?.send(JSON.stringify({ toolResponse: { functionResponses: responses } }));
    }
  }

  private async startMic(): Promise<void> {
    if (!this.stream) return;
    const ctx = new AudioContext({ sampleRate: MIC_RATE });
    const source = ctx.createMediaStreamSource(this.stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => {
      if (!this.ready || !this.isOpen() || this.muted) return;
      const data = float32ToPcm16Base64(e.inputBuffer.getChannelData(0));
      this.ws?.send(JSON.stringify({ realtimeInput: { audio: { mimeType: `audio/pcm;rate=${MIC_RATE}`, data } } }));
    };
    source.connect(proc);
    proc.connect(ctx.destination);
    this.micCtx = ctx;
    this.proc = proc;
  }

  private playChunk(samples: Float32Array): void {
    if (!this.playCtx) this.playCtx = new AudioContext({ sampleRate: SPEAKER_RATE });
    const ctx = this.playCtx;
    const buffer = ctx.createBuffer(1, samples.length, SPEAKER_RATE);
    buffer.getChannelData(0).set(samples);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.02, this.playAt);
    src.start(startAt);
    this.playAt = startAt + buffer.duration;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      if (this.sources.size === 0 && !this.stopped) this.setState("listening");
    };
  }

  private playingNow(): boolean {
    return this.sources.size > 0;
  }

  private stopPlayback(): void {
    for (const src of this.sources) {
      try { src.stop(); } catch { /* already ended */ }
    }
    this.sources.clear();
    this.playAt = 0;
  }
}
