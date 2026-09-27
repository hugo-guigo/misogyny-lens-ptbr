// Loads the quantized BERTimbau from the Hugging Face Hub and runs it in the browser
// with ONNX Runtime Web (through Transformers.js).
import { AutoTokenizer, AutoModelForSequenceClassification } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

export const MODEL_ID = "hugo-guigo/bertimbau-misoginia-ptbr";
export const MAX_LEN = 64;

let tokenizer = null, model = null;

export async function load(onProgress) {
  const files = {};
  const progress_callback = (p) => {
    if (p.status === "progress" && p.total) {
      files[p.file] = { loaded: p.loaded, total: p.total };
      const all = Object.values(files);
      onProgress(all.reduce((s, f) => s + f.loaded, 0) / all.reduce((s, f) => s + f.total, 0));
    }
  };
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, { dtype: "q8", device: "wasm", progress_callback });
  onProgress(1);
}

// Probability of "misogyny traits" for each text.
export async function proba(texts) {
  const inputs = tokenizer(texts, { padding: true, truncation: true, max_length: MAX_LEN });
  const { logits } = await model(inputs);
  const out = [], d = logits.data;
  for (let i = 0; i < texts.length; i++) {
    const a = d[2 * i], b = d[2 * i + 1], m = Math.max(a, b);
    const ea = Math.exp(a - m), eb = Math.exp(b - m);
    out.push(eb / (ea + eb));
  }
  return out;
}

// Occlusion: the text plus one copy without each word, in a single batch.
const WORD = /[\p{L}\p{N}_]+/gu;
export async function explain(text, maxWords = 30) {
  const words = [...text.matchAll(WORD)].slice(0, maxWords).map((m) => ({ start: m.index, end: m.index + m[0].length }));
  const variants = [text, ...words.map((w) => text.slice(0, w.start) + text.slice(w.end))];
  const p = await proba(variants);
  return {
    probability: p[0],
    words: words.map((w, i) => ({ ...w, text: text.slice(w.start, w.end), importance: p[0] - p[i + 1] })),
  };
}
