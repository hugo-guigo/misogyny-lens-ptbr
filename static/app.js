(function () {
  var $ = function (id) { return document.getElementById(id); };
  var root = document.documentElement;
  var text = $("text");
  var timer = null, seq = 0;
  var lastResult = null, lastRtt = 0, lastCard = null, health = {};
  var view = "bert";

  var EXAMPLES = [
    "Lugar de mulher é na cozinha",
    "Mulher no volante, perigo constante",
    "Ela é uma cientista brilhante e merece o prêmio",
    "O juiz roubou o jogo de ontem, que vergonha",
    "Essas feministas só sabem reclamar"
  ];

  function isEn() { return root.lang === "en"; }
  function t(pt, en) { return isEn() ? en : pt; }
  function fmt(x, d) { var s = x.toFixed(d); return isEn() ? s : s.replace(".", ","); }
  function pct(x) { return fmt(x * 100, 1) + "%"; }
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // ---------- Controls ----------
  $("lang").addEventListener("click", function () {
    root.lang = isEn() ? "pt-BR" : "en";
    try { localStorage.setItem("lang", root.lang); } catch (e) {}
    renderCard(lastCard);
    if (lastResult) render(lastResult, lastRtt);
  });

  EXAMPLES.forEach(function (ex) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = ex;
    b.addEventListener("click", function () { text.value = ex; schedule(0); });
    $("examples").appendChild(b);
  });

  document.querySelectorAll(".tab").forEach(function (b) {
    b.addEventListener("click", function () { setView(b.dataset.view); });
  });

  function setView(v) {
    view = v;
    document.querySelectorAll(".tab").forEach(function (b) { b.classList.toggle("on", b.dataset.view === v); });
    document.body.classList.toggle("view-svm", v === "svm");
    if (lastResult) render(lastResult, lastRtt);
  }

  text.addEventListener("input", function () { schedule(300); });

  function schedule(delay) {
    $("count").textContent = text.value.length + "/1000";
    clearTimeout(timer);
    timer = setTimeout(analyze, delay);
  }

  // ---------- API ----------
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
    ["lens", "parts", "models"].forEach(function (id) { $(id).innerHTML = ""; });
    $("verdict").textContent = "…";
    $("verdict").className = "verdict";
    $("gauge").style.width = "0";
    $("latency").textContent = $("rtt").textContent = "–";
  }

  // ---------- Rendering ----------
  function render(res, rtt) {
    var yes = res.label === 1;
    $("verdict").textContent = yes ? t("Traços de misoginia", "Misogyny traits") : t("Sem traços de misoginia", "No misogyny traits");
    $("verdict").className = "verdict " + (yes ? "yes" : "no");
    $("verdict-model").textContent = res.verdict_model;

    // Gauge: probability around 50% when available, else the SVM score clipped to [-2, 2].
    var g = $("gauge"), frac;
    if (res.probability !== null) {
      frac = res.probability - 0.5;              // [-0.5, 0.5]
      $("gauge-mid-label").textContent = "50%";
    } else {
      frac = Math.max(-2, Math.min(2, res.svm_decision)) / 4;
      $("gauge-mid-label").textContent = "0";
    }
    g.style.width = (Math.abs(frac) * 100) + "%";
    g.style.left = frac >= 0 ? "50%" : (50 - Math.abs(frac) * 100) + "%";
    g.className = "gauge-fill " + (frac >= 0 ? "pos" : "neg");

    var rows = [];
    if (res.probability !== null && res.verdict_model === "ensemble") rows.push(["ensemble", pct(res.probability)]);
    if (res.bert_probability !== null) rows.push(["BERTimbau", pct(res.bert_probability)]);
    rows.push(["SVM", (res.svm_decision >= 0 ? "+" : "") + fmt(res.svm_decision, 2)]);
    $("models").innerHTML = rows.map(function (r) { return "<div><dt>" + r[0] + "</dt><dd>" + r[1] + "</dd></div>"; }).join("");

    $("latency").textContent = fmt(res.latency_ms, 0) + " ms";
    $("rtt").textContent = fmt(rtt, 0) + " ms";

    var useBert = view === "bert" && res.bert_words;
    document.querySelector('.tab[data-view="bert"]').disabled = !res.bert_words;
    if (useBert) renderBert(res); else renderSvm(res);
  }

  function highlight(spans) {
    // spans: [{start, end, value, cls, tip}] sorted by start
    var src = text.value, html = "", pos = 0;
    var maxAbs = spans.reduce(function (m, s) { return Math.max(m, Math.abs(s.value)); }, 0.05);
    spans.forEach(function (s) {
      if (s.start < pos) return;
      html += escapeHtml(src.slice(pos, s.start));
      var a = Math.min(1, Math.abs(s.value) / maxAbs), cls = "tk " + (s.cls || "");
      var style = "";
      if (s.value) {
        cls += " on" + (a > 0.45 ? " strong" : "");
        style = "--a:" + (0.15 + 0.75 * a).toFixed(2) + ";--c:" + (s.value > 0 ? "255,106,43" : "200,245,58");
      }
      html += '<span class="' + cls + '" style="' + style + '" title="' + escapeHtml(s.tip) + '">' + escapeHtml(src.slice(s.start, s.end)) + "</span>";
      pos = s.end;
    });
    $("lens").innerHTML = html + escapeHtml(src.slice(pos));
  }

  function bars(items) {
    var scale = items.reduce(function (m, r) { return Math.max(m, Math.abs(r.value)); }, 0.05);
    $("parts").innerHTML = items.map(function (r) {
      var w = Math.abs(r.value) / scale * 50;
      return '<li class="' + (r.special ? "special" : "") + '"><span class="pn">' + escapeHtml(r.name) + '</span>' +
        '<span class="bar"><span class="b ' + (r.value >= 0 ? "pos" : "neg") + '" style="width:' + w + "%;" + (r.value >= 0 ? "left:50%" : "right:50%") + '"></span></span>' +
        '<span class="pv">' + (r.value >= 0 ? "+" : "") + fmt(r.value, 3) + "</span></li>";
    }).join("");
  }

  function renderBert(res) {
    highlight(res.bert_words.map(function (w) {
      return { start: w.start, end: w.end, value: w.importance,
               tip: t("sem esta palavra, a probabilidade muda ", "without this word, probability changes ") + (w.importance >= 0 ? "−" : "+") + pct(Math.abs(w.importance)) };
    }));
    $("lens-note").textContent = t(
      "Oclusão: o BERT roda de novo sem cada palavra. Laranja = sem ela, a probabilidade de traços cai. É uma estimativa de importância, não uma soma exata.",
      "Occlusion: BERT runs again without each word. Orange = without it, the probability of traits drops. It estimates importance; it is not an exact sum.");
    $("parts-title").textContent = t("Palavras que mais pesaram (BERT)", "Most influential words (BERT)");
    var words = res.bert_words.slice().sort(function (a, b) { return Math.abs(b.importance) - Math.abs(a.importance); }).slice(0, 7);
    bars(words.map(function (w) { return { name: w.text, value: w.importance }; }));
    $("parts-note").textContent = t("Variação da probabilidade de traços ao remover cada palavra.", "Change in the probability of traits when each word is removed.");
  }

  function renderSvm(res) {
    highlight(res.tokens.map(function (tk) {
      var cls = !tk.lemma ? "stop" : !tk.in_vocabulary ? "oov" : "";
      if (tk.lexicon === "misogino") cls += " lexi";
      return { start: tk.start, end: tk.end, value: tk.contribution, cls: cls,
               tip: tk.lemma ? (t("lema", "lemma") + ": " + tk.lemma + " · " + (tk.contribution >= 0 ? "+" : "") + fmt(tk.contribution, 3)) : t("ignorada (stopword ou símbolo)", "ignored (stopword or symbol)") };
    }));
    $("lens-note").textContent = t(
      "SVM linear: a pontuação é exatamente a soma das palavras + léxico + viés.",
      "Linear SVM: the score is exactly the sum of words + lexicon + bias.");
    $("parts-title").textContent = t("De onde vem a pontuação do SVM", "Where the SVM score comes from");
    var seen = {}, words = [];
    res.tokens.forEach(function (tk) {
      if (!tk.lemma || !tk.contribution) return;
      if (seen[tk.lemma]) { seen[tk.lemma].value += tk.contribution; return; }
      seen[tk.lemma] = { name: tk.lemma, value: tk.contribution };
      words.push(seen[tk.lemma]);
    });
    words.sort(function (a, b) { return Math.abs(b.value) - Math.abs(a.value); });
    bars(words.slice(0, 6).concat([
      { name: t("léxico (8 atributos)", "lexicon (8 features)"), value: res.lexicon_contribution, special: true },
      { name: t("viés do modelo", "model bias"), value: res.bias, special: true }
    ]));
    $("parts-note").textContent = t("Acima de 0, o SVM marca \"com traços\".", "Above 0, the SVM says \"traits\".");
  }

  // ---------- Model card ----------
  function renderCard(c) {
    if (!c || !c.svm) return;
    var e = c.ensemble, b = c.bert, s = c.svm;
    var head = e ? e.test.ensemble : b ? b.test : s.test;
    var boot = e ? e.test_ensemble_bootstrap : null;
    var items = [
      [t("Acurácia (teste)", "Accuracy (test)"), pct(head.accuracy), boot ? t("IC 95% ", "95% CI ") + pct(boot.accuracy_ci95[0]) + " – " + pct(boot.accuracy_ci95[1]) : ""],
      ["F1 " + t("(teste)", "(test)"), fmt(head.f1, 3), boot ? t("IC 95% ", "95% CI ") + fmt(boot.f1_ci95[0], 3) + " – " + fmt(boot.f1_ci95[1], 3) : ""],
      [t("Precisão", "Precision"), pct(head.precision), t("quando marca, quase sempre acerta", "when it flags, it is almost always right")],
      ["Recall", pct(head.recall), t("deixa passar ", "misses ") + pct(1 - head.recall) + t(" dos casos", " of cases")],
      [t("Vazamento removido", "Leak removed"), String(s.leaked_rows_removed), t("linhas do treino iguais ao teste", "training rows equal to test")],
      [t("Teste", "Test set"), String(s.test_docs), t("frases, usadas uma vez", "sentences, used once")]
    ];
    $("card").innerHTML = items.map(function (it) {
      return '<div class="stat"><p class="k mono">' + it[0] + '</p><p class="v">' + it[1] + '</p><p class="d">' + it[2] + "</p></div>";
    }).join("");

    var rows = [["SVM (TF-IDF + " + t("léxico", "lexicon") + ")", s.test]];
    if (b) rows.push(["BERTimbau fine-tuned", b.test]);
    if (e) rows.push(["Ensemble", e.test.ensemble]);
    $("cmp").innerHTML = "<thead><tr><th>" + t("Modelo", "Model") + "</th><th>" + t("Acurácia", "Accuracy") + "</th><th>F1</th><th>" + t("Precisão", "Precision") + "</th><th>Recall</th></tr></thead><tbody>" +
      rows.map(function (r) {
        return "<tr><td>" + r[0] + "</td><td>" + pct(r[1].accuracy) + "</td><td>" + fmt(r[1].f1, 3) + "</td><td>" + pct(r[1].precision) + "</td><td>" + pct(r[1].recall) + "</td></tr>";
      }).join("") + "</tbody>";
  }

  fetch("/api/health").then(function (r) { return r.json(); }).then(function (h) {
    health = h;
    setView(h.bert_loaded ? "bert" : "svm");
    schedule(0);
  });
  fetch("/api/model-card").then(function (r) { return r.json(); }).then(function (m) { lastCard = m; renderCard(m); });
})();
