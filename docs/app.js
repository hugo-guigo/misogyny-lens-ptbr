import { load, explain } from "./model.js";

const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const text = $("text");
let timer = null, running = false, pending = false, ready = false, last = null;

const EXAMPLES = [
  "Lugar de mulher é na cozinha",
  "Mulher no volante, perigo constante",
  "Ela é uma cientista brilhante e merece o prêmio",
  "O juiz roubou o jogo de ontem, que vergonha",
  "Essas feministas só sabem reclamar",
];

const isEn = () => root.lang === "en";
const t = (pt, en) => (isEn() ? en : pt);
const fmt = (x, d) => { const s = x.toFixed(d); return isEn() ? s : s.replace(".", ","); };
const pct = (x) => fmt(x * 100, 1) + "%";
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

$("lang").addEventListener("click", () => {
  root.lang = isEn() ? "pt-BR" : "en";
  try { localStorage.setItem("lang", root.lang); } catch (e) {}
  if (last) render(last.res, last.ms);
});

for (const ex of EXAMPLES) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = ex;
  b.addEventListener("click", () => { text.value = ex; schedule(0); });
  $("examples").appendChild(b);
}

text.addEventListener("input", () => schedule(350));

function schedule(delay) {
  $("count").textContent = text.value.length + "/1000";
  clearTimeout(timer);
  timer = setTimeout(run, delay);
}

// One inference at a time; if the text changes meanwhile, run again with the newest text.
async function run() {
  if (!ready) return;
  if (running) { pending = true; return; }
  const value = text.value;
  if (!value.trim()) { $("lens").innerHTML = ""; $("parts").innerHTML = ""; $("verdict").textContent = "…"; return; }
  running = true;
  const t0 = performance.now();
  const res = await explain(value);
  const ms = performance.now() - t0;
  running = false;
  if (value === text.value) { last = { res, ms }; render(res, ms); }
  if (pending) { pending = false; run(); }
}

function render(res, ms) {
  const yes = res.probability >= 0.5;
  $("verdict").textContent = yes ? t("Traços de misoginia", "Misogyny traits") : t("Sem traços de misoginia", "No misogyny traits");
  $("verdict").className = "verdict " + (yes ? "yes" : "no");
  const frac = res.probability - 0.5, g = $("gauge");
  g.style.width = Math.abs(frac) * 100 + "%";
  g.style.left = frac >= 0 ? "50%" : 50 - Math.abs(frac) * 100 + "%";
  g.className = "gauge-fill " + (frac >= 0 ? "pos" : "neg");
  $("prob").textContent = pct(res.probability);
  $("latency").textContent = fmt(ms, 0) + " ms";
  $("backend").textContent = t(`${res.words.length + 1} passagens do modelo num lote · WASM, na sua máquina`,
                               `${res.words.length + 1} model passes in one batch · WASM, on your machine`);

  const src = text.value;
  const maxAbs = res.words.reduce((m, w) => Math.max(m, Math.abs(w.importance)), 0.05);
  let html = "", pos = 0;
  for (const w of res.words) {
    html += esc(src.slice(pos, w.start));
    const a = Math.min(1, Math.abs(w.importance) / maxAbs);
    const cls = "tk on" + (a > 0.45 ? " strong" : "");
    const style = `--a:${(0.15 + 0.75 * a).toFixed(2)};--c:${w.importance > 0 ? "255,106,43" : "200,245,58"}`;
    const tip = t("sem esta palavra: ", "without this word: ") + pct(res.probability - w.importance);
    html += `<span class="${cls}" style="${style}" title="${esc(tip)}">${esc(w.text)}</span>`;
    pos = w.end;
  }
  $("lens").innerHTML = html + esc(src.slice(pos));

  const top = [...res.words].sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance)).slice(0, 7);
  const scale = top.reduce((m, w) => Math.max(m, Math.abs(w.importance)), 0.05);
  $("parts").innerHTML = top.map((w) => {
    const width = Math.abs(w.importance) / scale * 50;
    const side = w.importance >= 0 ? "left:50%" : "right:50%";
    return `<li><span class="pn">${esc(w.text)}</span><span class="bar"><span class="b ${w.importance >= 0 ? "pos" : "neg"}" style="width:${width}%;${side}"></span></span><span class="pv">${w.importance >= 0 ? "+" : ""}${fmt(w.importance * 100, 1)}</span></li>`;
  }).join("");
}

load((f) => { $("loader-bar").style.width = Math.round(f * 100) + "%"; })
  .then(() => { ready = true; $("loader").classList.add("done"); schedule(0); })
  .catch((e) => {
    $("loader-text").textContent = t("Não consegui carregar o modelo neste navegador: ", "Could not load the model in this browser: ") + e.message;
  });
