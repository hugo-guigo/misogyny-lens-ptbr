(function () {
  var $ = function (id) { return document.getElementById(id); };
  var root = document.documentElement;
  var text = $("text");
  var timer = null, seq = 0;

  var EXAMPLES = [
    "Lugar de mulher é na cozinha",
    "Mulher no volante, perigo constante",
    "Ela é uma cientista brilhante e merece o prêmio",
    "O juiz roubou o jogo de ontem, que vergonha",
    "Essas feministas só sabem reclamar"
  ];

  function isEn() { return root.lang === "en"; }
  function t(pt, en) { return isEn() ? en : pt; }

  // ---------- Language ----------
  $("lang").addEventListener("click", function () {
    root.lang = isEn() ? "pt-BR" : "en";
    try { localStorage.setItem("lang", root.lang); } catch (e) {}
    renderCard(lastCard);
    if (lastResult) render(lastResult, lastRtt);
  });

  // ---------- Examples ----------
  EXAMPLES.forEach(function (ex) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = ex;
    b.addEventListener("click", function () { text.value = ex; schedule(0); });
    $("examples").appendChild(b);
  });

  text.addEventListener("input", function () { schedule(250); });

  function schedule(delay) {
    $("count").textContent = text.value.length + "/1000";
    clearTimeout(timer);
    timer = setTimeout(analyze, delay);
  }

  // ---------- API ----------
  var lastResult = null, lastRtt = 0, lastCard = null;

  function analyze() {
    var value = text.value;
    if (!value.trim()) { clearView(); return; }
    var mine = ++seq, t0 = performance.now();
    fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: value })
    })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (res) {
        if (mine !== seq) return;  // a newer request is on its way
        lastResult = res;
        lastRtt = performance.now() - t0;
        render(res, lastRtt);
      })
      .catch(function () {
        if (mine !== seq) return;
        $("verdict").textContent = t("Não consegui falar com a API.", "Could not reach the API.");
        $("verdict").className = "verdict";
      });
  }

  function clearView() {
    lastResult = null;
    $("lens").innerHTML = "";
    $("parts").innerHTML = "";
    $("verdict").textContent = "…";
    $("verdict").className = "verdict";
    $("gauge").style.width = "0";
    ["score", "latency", "rtt"].forEach(function (id) { $(id).textContent = "–"; });
  }

  // ---------- Rendering ----------
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function render(res, rtt) {
    var yes = res.label === 1;
    $("verdict").textContent = yes ? t("Traços de misoginia", "Misogyny traits") : t("Sem traços de misoginia", "No misogyny traits");
    $("verdict").className = "verdict " + (yes ? "yes" : "no");

    // Gauge: score clipped to [-2, 2], filled from the center.
    var s = Math.max(-2, Math.min(2, res.decision)), g = $("gauge");
    g.style.width = (Math.abs(s) / 4 * 100) + "%";
    g.style.left = s >= 0 ? "50%" : (50 - Math.abs(s) / 4 * 100) + "%";
    g.className = "gauge-fill " + (s >= 0 ? "pos" : "neg");

    $("score").textContent = (res.decision >= 0 ? "+" : "") + res.decision.toFixed(3);
    $("latency").textContent = res.latency_ms.toFixed(1) + " ms";
    $("rtt").textContent = rtt.toFixed(0) + " ms";

    // Highlighted text, rebuilt from the original string using token offsets.
    var src = text.value, html = "", pos = 0;
    var maxAbs = res.tokens.reduce(function (m, tk) { return Math.max(m, Math.abs(tk.contribution)); }, 0.05);
    res.tokens.forEach(function (tk) {
      if (tk.start < pos) return;
      html += escapeHtml(src.slice(pos, tk.start));
      var c = tk.contribution, a = Math.min(1, Math.abs(c) / maxAbs);
      var cls = "tk";
      if (!tk.lemma) cls += " stop";
      else if (!tk.in_vocabulary) cls += " oov";
      if (tk.lexicon === "misogino") cls += " lexi";
      var style = c > 0 ? "--a:" + (0.15 + 0.75 * a).toFixed(2) + ";--c:255,106,43"
                : c < 0 ? "--a:" + (0.15 + 0.75 * a).toFixed(2) + ";--c:200,245,58" : "";
      var tip = tk.lemma ? (t("lema", "lemma") + ": " + tk.lemma + " · " + (c >= 0 ? "+" : "") + c.toFixed(3)) : t("ignorada (stopword ou símbolo)", "ignored (stopword or symbol)");
      if (c && a > 0.45) cls += " strong";
      html += '<span class="' + cls + (c ? " on" : "") + '" style="' + style + '" title="' + escapeHtml(tip) + '">' + escapeHtml(src.slice(tk.start, tk.end)) + "</span>";
      pos = tk.end;
    });
    html += escapeHtml(src.slice(pos));
    $("lens").innerHTML = html;

    // Breakdown: top words by |contribution|, then lexicon and bias.
    var seen = {}, words = [];
    res.tokens.forEach(function (tk) {
      if (!tk.lemma || !tk.contribution) return;
      if (seen[tk.lemma]) { seen[tk.lemma].value += tk.contribution; return; }
      seen[tk.lemma] = { name: tk.lemma, value: tk.contribution };
      words.push(seen[tk.lemma]);
    });
    words.sort(function (a, b) { return Math.abs(b.value) - Math.abs(a.value); });
    var rows = words.slice(0, 6).concat([
      { name: t("léxico (8 atributos)", "lexicon (8 features)"), value: res.lexicon_contribution, special: true },
      { name: t("viés do modelo", "model bias"), value: res.bias, special: true }
    ]);
    var scale = rows.reduce(function (m, r) { return Math.max(m, Math.abs(r.value)); }, 0.1);
    $("parts").innerHTML = rows.map(function (r) {
      var w = Math.abs(r.value) / scale * 50;
      return '<li class="' + (r.special ? "special" : "") + '"><span class="pn">' + escapeHtml(r.name) + '</span>' +
        '<span class="bar"><span class="b ' + (r.value >= 0 ? "pos" : "neg") + '" style="width:' + w + "%;" + (r.value >= 0 ? "left:50%" : "right:50%") + '"></span></span>' +
        '<span class="pv">' + (r.value >= 0 ? "+" : "") + r.value.toFixed(3) + "</span></li>";
    }).join("");
  }

  // ---------- Model card ----------
  function pct(x) { var s = (x * 100).toFixed(1) + "%"; return isEn() ? s : s.replace(".", ","); }
  function num(x) { return isEn() ? x.toFixed(3) : x.toFixed(3).replace(".", ","); }

  function renderCard(m) {
    if (!m) return;
    var b = m.bootstrap, L = m.test_lexicon, B = m.test_baseline_tfidf_only;
    var items = [
      ["F1 " + t("(teste)", "(test)"), num(L.f1), t("IC 95% ", "95% CI ") + num(b.f1_ci_lexicon[0]) + " – " + num(b.f1_ci_lexicon[1])],
      [t("Precisão", "Precision"), pct(L.precision), t("quando marca, costuma acertar", "when it flags, it is usually right")],
      ["Recall", pct(L.recall), t("deixa passar ", "misses ") + pct(1 - L.recall) + t(" dos casos", " of cases")],
      [t("Ganho do léxico", "Lexicon gain"), (b.delta_mean >= 0 ? "+" : "") + num(b.delta_mean), t("F1 vs TF-IDF puro · p = ", "F1 vs plain TF-IDF · p = ") + num(b.p_one_sided) + t(" (não significativo)", " (not significant)")],
      [t("Vazamento removido", "Leak removed"), String(m.leaked_rows_removed), t("linhas do treino iguais ao teste", "training rows equal to test")],
      [t("Treino", "Training"), m.train_docs.toLocaleString(isEn() ? "en" : "pt-BR"), t("documentos · ", "documents · ") + pct(m.train_positive_rate) + t(" positivos", " positive")]
    ];
    $("card").innerHTML = items.map(function (it) {
      return '<div class="stat"><p class="k mono">' + it[0] + '</p><p class="v">' + it[1] + '</p><p class="d">' + it[2] + "</p></div>";
    }).join("");
  }

  fetch("/api/model-card").then(function (r) { return r.json(); }).then(function (m) { lastCard = m; renderCard(m); });

  schedule(0);
})();
