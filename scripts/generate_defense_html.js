/**
 * NBU Thesis Defense Presentation Generator — HTML
 * Generates defense_presentation.html: self-contained, keyboard-navigable, base64 images.
 * 20 slides for a 15-20 minute defense.
 *
 * Usage: node scripts/generate_defense_html.js
 * Output: defense_presentation.html in project root
 */

const fs = require("fs");
const path = require("path");

// ─── CONSTANTS ───────────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, "..");
const RESULTS = path.join(ROOT, "results");

// ─── IMAGE LOADING ───────────────────────────────────────────────────────────

function loadBase64(filename) {
  const p = path.join(RESULTS, filename);
  if (fs.existsSync(p)) {
    return "data:image/png;base64," + fs.readFileSync(p).toString("base64");
  }
  console.warn(`WARNING: Missing image ${filename}`);
  return "";
}

// ─── BUILD HTML ──────────────────────────────────────────────────────────────

function main() {
  console.log("Generating defense_presentation.html (20 slides)...");

  const images = {
    heatmap:       loadBase64("co_occurrence_heatmap.png"),
    classicBar:    loadBase64("classic_ml_f1_barchart.png"),
    grand:         loadBase64("fig_grand_comparison.png"),
    categoryHeat:  loadBase64("fig_category_heatmap.png"),
    scale:         loadBase64("fig_scale_effect.png"),
    parentVsSub:   loadBase64("fig_parent_vs_sub.png"),
    cost:          loadBase64("fig_cost_vs_performance.png"),
  };

  const html = `<!DOCTYPE html>
<html lang="bg">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Защита на магистърска теза — Жарко Рашев — НБУ 2026</title>
<style>
  /* ─── RESET & BASE ─── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; background: #0D1B2A; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }

  /* ─── COLORS ─── */
  :root {
    --navy: #0D1B2A; --steel: #1B4965; --sky: #5FA8D3;
    --green: #2CA02C; --blue: #1F77B4; --orange: #FF7F0E; --red: #D62728;
    --bg: #F8F9FA; --text: #1E293B; --muted: #64748B;
    --white: #FFFFFF; --light-gray: #E2E8F0;
  }

  /* ─── SLIDES ─── */
  .slide {
    position: absolute; top: 0; left: 0;
    width: 100vw; height: 100vh;
    display: flex; flex-direction: column;
    opacity: 0; pointer-events: none;
    transition: opacity 0.35s ease;
  }
  .slide.active { opacity: 1; pointer-events: auto; }
  .slide.dark { background: var(--navy); color: var(--white); }
  .slide.light { background: var(--bg); color: var(--text); }

  /* ─── HEADER BAR ─── */
  .header-bar {
    background: var(--steel); padding: 16px 40px; flex-shrink: 0;
  }
  .header-bar h2 {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 28px; color: var(--white); font-weight: 700;
  }

  /* ─── CONTENT AREA ─── */
  .content { flex: 1; padding: 24px 40px; overflow: hidden; display: flex; flex-direction: column; }
  .content.centered { align-items: center; justify-content: center; text-align: center; }

  /* ─── TYPOGRAPHY ─── */
  h1 { font-family: Georgia, serif; }
  h2, h3 { font-family: Georgia, serif; }
  .subtitle { color: var(--sky); font-size: 20px; margin-top: 12px; }
  .caption { color: var(--muted); font-style: italic; font-size: 13px; text-align: center; margin-top: 8px; }
  .footnote { color: var(--muted); font-style: italic; font-size: 13px; text-align: center; padding: 8px; }

  /* ─── TWO COLUMNS ─── */
  .two-col { display: flex; gap: 32px; flex: 1; min-height: 0; }
  .col { flex: 1; display: flex; flex-direction: column; justify-content: center; min-width: 0; }

  /* ─── STAT CALLOUTS ─── */
  .stat-card {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 16px; text-align: center; margin: 8px 0;
  }
  .stat-card .value { font-family: Consolas, monospace; font-size: 36px; font-weight: 700; }
  .stat-card .label { color: var(--muted); font-size: 13px; margin-top: 4px; }

  /* ─── CARDS ─── */
  .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; flex: 1; }
  .card {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 20px 20px 20px 26px; position: relative;
  }
  .card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 5px; border-radius: 8px 0 0 8px;
  }
  .card.blue::before { background: var(--blue); }
  .card.orange::before { background: var(--orange); }
  .card.red::before { background: var(--red); }
  .card.green::before { background: var(--green); }
  .card.sky::before { background: var(--sky); }
  .card.steel::before { background: var(--steel); }
  .card.muted-accent::before { background: var(--muted); }
  .card h3 { font-size: 17px; margin-bottom: 8px; }
  .card p { font-size: 14px; color: var(--muted); line-height: 1.5; }

  /* ─── FINDING CARDS ─── */
  .finding {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 10px 16px 10px 22px; position: relative; margin: 4px 0;
  }
  .finding::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 5px; border-radius: 8px 0 0 8px;
  }
  .finding h3 { font-size: 14px; margin-bottom: 3px; }
  .finding p { font-size: 12px; color: var(--muted); line-height: 1.4; }

  /* ─── TABLE ─── */
  table { width: 100%; border-collapse: collapse; margin: 12px 0; }
  thead th {
    background: var(--steel); color: var(--white);
    padding: 8px 10px; font-size: 13px; text-align: center;
  }
  tbody td {
    padding: 7px 10px; text-align: center; font-size: 12px; border-bottom: 1px solid var(--light-gray);
  }
  tbody tr:nth-child(even) { background: var(--light-gray); }
  tbody td:first-child { font-family: Georgia, serif; font-weight: 600; text-align: left; }

  /* ─── IMAGES ─── */
  .fig-container { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
  .fig-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .fig-with-side { display: flex; gap: 24px; flex: 1; min-height: 0; }
  .fig-main { flex: 2; display: flex; align-items: center; justify-content: center; min-height: 0; }
  .fig-main img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .fig-side { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 12px; }

  /* ─── TAKEAWAY BAR ─── */
  .takeaway {
    background: var(--white); border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    padding: 12px 20px; text-align: center; font-size: 14px;
    color: var(--text); flex-shrink: 0; margin-top: 8px;
  }

  /* ─── PIPELINE (NEW) ─── */
  .pipeline { display: flex; align-items: center; justify-content: center; gap: 0; margin: 16px 0; }
  .pipe-step {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    padding: 14px 16px; text-align: center; min-width: 130px;
    border-top: 4px solid var(--steel);
  }
  .pipe-step h4 { font-size: 14px; margin-bottom: 4px; }
  .pipe-step p { font-size: 11px; color: var(--muted); }
  .pipe-arrow { font-size: 28px; color: var(--muted); margin: 0 6px; }

  /* ─── VS COMPARE (NEW) ─── */
  .vs-compare { display: flex; gap: 24px; flex: 1; min-height: 0; }
  .vs-box { flex: 1; display: flex; flex-direction: column; }

  /* ─── CODE BLOCK (NEW) ─── */
  .code-block {
    background: #F1F5F9; border: 1px solid var(--light-gray); border-radius: 6px;
    padding: 12px 16px; font-family: Consolas, monospace; font-size: 11px;
    line-height: 1.4; white-space: pre-wrap; overflow: auto; flex: 1;
    color: var(--text);
  }
  .code-label { font-weight: 700; font-size: 13px; margin-bottom: 6px; }

  /* ─── MINI TABLE (NEW) ─── */
  .mini-table { margin: 8px 0; }
  .mini-table th { padding: 6px 8px; font-size: 11px; }
  .mini-table td { padding: 5px 8px; font-size: 11px; }

  /* ─── DIAGRAM BOX (NEW) ─── */
  .diagram-box {
    border: 2px solid var(--steel); border-radius: 8px;
    padding: 12px 16px; background: var(--white);
    border-top: 4px solid var(--steel);
  }
  .diagram-box h4 { font-family: Georgia, serif; font-size: 14px; margin-bottom: 6px; }
  .diagram-box p { font-size: 12px; color: var(--text); line-height: 1.4; }
  .diagram-box.accent-blue { border-color: var(--blue); border-top-color: var(--blue); }
  .diagram-box.accent-blue h4 { color: var(--blue); }
  .diagram-box.accent-sky { border-color: var(--sky); border-top-color: var(--sky); }
  .diagram-box.accent-sky h4 { color: var(--sky); }
  .diagram-box.accent-green { border-color: var(--green); border-top-color: var(--green); }
  .diagram-box.accent-green h4 { color: var(--green); }
  .diagram-box.accent-orange { border-color: var(--orange); border-top-color: var(--orange); }
  .diagram-box.accent-orange h4 { color: var(--orange); }
  .diagram-box.accent-red { border-color: var(--red); border-top-color: var(--red); }
  .diagram-box.accent-red h4 { color: var(--red); }
  .diagram-box.accent-muted { border-color: var(--muted); border-top-color: var(--muted); }
  .diagram-box.accent-muted h4 { color: var(--muted); }

  /* ─── REASON CARD (NEW) ─── */
  .reason-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; flex: 1; }

  /* ─── RECOMMENDATION BOX (NEW) ─── */
  .recommendation-box {
    background: #DCFCE7; border-radius: 8px; padding: 14px 20px;
    font-size: 13px; color: var(--text); text-align: center; margin-top: 8px;
  }

  /* ─── WARNING BOX ─── */
  .warning-box {
    background: #FFF3CD; border-radius: 6px; padding: 12px 18px;
    font-size: 12px; color: var(--text);
  }

  /* ─── STAT ROW ─── */
  .stat-row { display: flex; gap: 12px; }

  /* ─── KEY STATS ─── */
  .key-stats { display: flex; flex-direction: column; gap: 10px; }
  .key-stat { display: flex; align-items: baseline; gap: 12px; }
  .key-stat .ks-val { font-family: Consolas, monospace; font-size: 16px; font-weight: 700; color: var(--sky); min-width: 90px; text-align: right; }
  .key-stat .ks-label { font-size: 12px; color: var(--text); }

  /* ─── HIERARCHY ─── */
  .hierarchy { display: flex; flex-direction: column; gap: 10px; }
  .hierarchy-item {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    padding: 10px 14px 10px 44px; position: relative;
  }
  .hierarchy-item .num {
    position: absolute; left: 8px; top: 50%; transform: translateY(-50%);
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 14px;
  }
  .hierarchy-item h4 { font-size: 13px; margin-bottom: 2px; }
  .hierarchy-item p { font-size: 11px; color: var(--muted); }

  /* ─── THREE COL ─── */
  .three-col { display: flex; gap: 16px; }
  .three-col > div { flex: 1; }

  /* ─── CONTROLS ─── */
  .slide-counter {
    position: fixed; bottom: 16px; right: 24px;
    font-family: Consolas, monospace; font-size: 14px; color: rgba(255,255,255,0.5);
    z-index: 100; pointer-events: none;
  }
  .progress-bar {
    position: fixed; bottom: 0; left: 0; height: 3px;
    background: var(--sky); z-index: 100;
    transition: width 0.35s ease;
  }

  /* ─── OVERVIEW MODE ─── */
  .overview .slide {
    position: relative !important; display: flex !important;
    opacity: 1 !important; pointer-events: auto !important;
    width: 280px; height: 157px; border-radius: 8px;
    margin: 8px; cursor: pointer; overflow: hidden;
    transform: scale(1); transition: transform 0.2s, box-shadow 0.2s;
    flex-shrink: 0; border: 2px solid transparent;
  }
  .overview .slide:hover { transform: scale(1.05); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .overview .slide.active { border-color: var(--sky); }
  .overview { display: flex; flex-wrap: wrap; justify-content: center; align-items: center;
    align-content: center; height: 100vh; overflow: auto; padding: 20px; background: #111; }
  .overview .slide * { transform: scale(0.21); transform-origin: top left; pointer-events: none; }
  .overview .slide-counter, .overview .progress-bar { display: none; }

  /* ─── FUTURE WORK ─── */
  .future-list { list-style: disc; padding-left: 24px; text-align: left; display: inline-block; }
  .future-list li { font-size: 15px; margin: 7px 0; color: var(--white); }

  /* ─── RESPONSIVE ─── */
  @media (max-width: 900px) {
    .two-col, .vs-compare, .fig-with-side, .three-col { flex-direction: column; }
    .card-grid, .reason-grid { grid-template-columns: 1fr; }
    .pipeline { flex-wrap: wrap; }
  }

  /* ─── PRINT ─── */
  @media print {
    .slide { position: relative !important; opacity: 1 !important; pointer-events: auto !important;
      page-break-after: always; width: 100vw; height: 100vh; }
    .progress-bar, .slide-counter { display: none; }
  }
</style>
</head>
<body>

<!-- ═══ SLIDE 1: TITLE ═══ -->
<section class="slide dark active" data-slide="0">
  <div class="content centered">
    <p style="color:var(--sky); font-style:italic; font-size:17px;">Нов български университет</p>
    <p style="color:var(--muted); font-size:13px; margin-top:4px;">Департамент по информатика</p>
    <h1 style="font-size:34px; line-height:1.3; max-width:900px; margin-top:20px;">Мултимодално дълбоко обучение за<br>класификация на инциденти по ВПП</h1>
    <p class="subtitle" style="font-size:18px; margin-top:14px;">Multi-Label Classification of Aviation Safety Reports<br>Using LLMs and Classic ML</p>
    <hr style="border:none; border-top:2px solid var(--sky); width:300px; margin:24px auto;">
    <p style="color:var(--white); font-size:16px;">Зарко Рашев &nbsp; Ф№ F98363</p>
    <p style="color:var(--muted); font-size:14px; margin-top:6px;">Научен ръководител: доц. д-р Стоян Мишев</p>
    <p style="color:var(--muted); font-size:13px; margin-top:8px;">Магистърска теза &nbsp;|&nbsp; 2026</p>
  </div>
</section>

<!-- ═══ SLIDE 2: RESEARCH QUESTION ═══ -->
<section class="slide light" data-slide="1">
  <div class="header-bar"><h2>Изследователски въпрос</h2></div>
  <div class="content">
    <div class="two-col">
      <div class="col">
        <p style="font-family:Georgia,serif; font-size:18px; font-style:italic; line-height:1.5; margin-bottom:20px;">
          Могат ли големите езикови модели (LLM) да конкурират класическото машинно обучение при мултилейбълна класификация на авиационни доклади за безопасност?
        </p>
        <ol class="sub-questions" style="list-style:none; padding:0;">
          <li style="font-size:14px; margin:8px 0;">1. Как се сравняват zero-shot, few-shot и fine-tuned LLM подходи?</li>
          <li style="font-size:14px; margin:8px 0;">2. Каква е ролята на размера на модела (8B vs 675B)?</li>
          <li style="font-size:14px; margin:8px 0;">3. Кога taxonomy-enriched промптинг помага?</li>
          <li style="font-size:14px; margin:8px 0;">4. Какъв е компромисът между цена и качество?</li>
        </ol>
      </div>
      <div class="col" style="gap:12px; display:flex; flex-direction:column; justify-content:flex-start; padding-top:8px;">
        <div class="stat-row">
          <div class="stat-card" style="flex:1; border-top:4px solid var(--sky);">
            <div class="value" style="color:var(--sky);">172K</div>
            <div class="label">ASRS доклада</div>
          </div>
          <div class="stat-card" style="flex:1; border-top:4px solid var(--green);">
            <div class="value" style="color:var(--green);">13</div>
            <div class="label">категории</div>
          </div>
          <div class="stat-card" style="flex:1; border-top:4px solid var(--orange);">
            <div class="value" style="color:var(--orange);">22</div>
            <div class="label">експеримента</div>
          </div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--red);">
          <div class="value" style="color:var(--red);">78%</div>
          <div class="label">мулти-лейбъл доклади (2+ категории)</div>
        </div>
      </div>
    </div>
    <p class="footnote">NASA Aviation Safety Reporting System &nbsp;|&nbsp; 1988\u20132024 &nbsp;|&nbsp; Доброволни наративни доклади от пилоти и контрольори</p>
  </div>
</section>

<!-- ═══ SLIDE 3: DATASET ═══ -->
<section class="slide light" data-slide="2">
  <div class="header-bar"><h2>Данни \u2014 NASA ASRS</h2></div>
  <div class="content">
    <div class="two-col">
      <div class="col" style="align-items:center;">
        ${images.heatmap ? `<img src="${images.heatmap}" alt="Co-occurrence heatmap" style="max-height:70%; object-fit:contain;">` : '<p style="color:var(--muted);">[Image not found]</p>'}
        <p class="caption">Корелационна матрица на съвместно появяване</p>
      </div>
      <div class="col">
        <table class="mini-table">
          <thead><tr><th>Категория</th><th>%</th></tr></thead>
          <tbody>
            <tr><td>Deviation-Procedural</td><td>65.4%</td></tr>
            <tr><td>Aircraft Equipment Problem</td><td>28.6%</td></tr>
            <tr><td>Conflict</td><td>26.9%</td></tr>
            <tr><td>Inflight Event/Encounter</td><td>22.5%</td></tr>
            <tr><td>ATC Issue</td><td>17.1%</td></tr>
            <tr><td style="text-align:center; color:var(--muted);" colspan="2">\u2026</td></tr>
            <tr><td>Airspace Violation</td><td>4.0%</td></tr>
            <tr><td>Deviation-Speed</td><td>2.9%</td></tr>
            <tr><td>Ground Excursion</td><td>2.2%</td></tr>
          </tbody>
        </table>
        <div class="key-stats" style="margin-top:12px;">
          <div class="key-stat"><span class="ks-val">31,850 / 8,044</span><span class="ks-label">train / test разделение</span></div>
          <div class="key-stat"><span class="ks-val">30.3\u00d7</span><span class="ks-label">дисбаланс между категории</span></div>
          <div class="key-stat"><span class="ks-val">78%</span><span class="ks-label">доклади с 2+ категории</span></div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 4: MULTI-LABEL EXPLAINED (NEW) ═══ -->
<section class="slide light" data-slide="3">
  <div class="header-bar"><h2>Какво е мулти-лейбъл класификация?</h2></div>
  <div class="content">
    <div class="two-col" style="gap:24px;">
      <div class="col" style="gap:16px;">
        <div class="diagram-box accent-muted">
          <h4>Multi-Class (стандартна)</h4>
          <p>1 доклад \u2192 1 категория<br>Взаимно изключващи се класове<br>Пример: спам / не-спам</p>
        </div>
        <div class="diagram-box accent-sky">
          <h4>Multi-Label (нашата задача)</h4>
          <p>1 доклад \u2192 1\u201313 категории<br>78% от докладите имат 2+ лейбъла<br>Медиана: 2 категории на доклад</p>
        </div>
      </div>
      <div class="col">
        <div class="diagram-box accent-blue" style="flex:1;">
          <h4>Примерен ASRS доклад</h4>
          <p style="font-family:Consolas,monospace; font-size:11px; line-height:1.4; margin:8px 0;">
            "While descending through FL240, we received a TCAS RA for traffic at our 12 o'clock. We followed the RA guidance and deviated from our assigned altitude. ATC was notified immediately but there was a delay in acknowledgment..."
          </p>
          <p style="margin-top:8px;"><strong>\u2192 Conflict<br>\u2192 Deviation - Altitude<br>\u2192 ATC Issue</strong></p>
        </div>
      </div>
    </div>
    <div class="three-col" style="margin-top:12px; flex-shrink:0;">
      <div class="diagram-box accent-green">
        <h4>Macro-F1</h4>
        <p>Средно F1 на 13 категории. Третира всяка еднакво. Чувствителна към редки класове.</p>
      </div>
      <div class="diagram-box accent-orange">
        <h4>Micro-F1</h4>
        <p>Глобално F1 от всички предикции. Претегля по брой примери. По-високо при чести категории.</p>
      </div>
      <div class="diagram-box accent-sky">
        <h4>Macro-AUC (ROC)</h4>
        <p>Качество на ранжирането. Независимо от прага. 1.0 = перфектно, 0.5 = случайно.</p>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 5: FOUR APPROACHES ═══ -->
<section class="slide light" data-slide="4">
  <div class="header-bar"><h2>Четири подхода за сравнение</h2></div>
  <div class="content">
    <div class="card-grid">
      <div class="card blue">
        <h3>1. Zero-Shot LLM</h3>
        <p>Директна класификация без примери.<br>Моделът разчита на предварително знание.<br>Тестван: Qwen3-8B, Mistral Large 3, DeepSeek V3.2</p>
      </div>
      <div class="card orange">
        <h3>2. Few-Shot LLM</h3>
        <p>3 примера на категория (39 общо) в промпта.<br>Подбрани: кратки, еднолейбълни от train set.<br>Тестван: Qwen3-8B, Mistral Large 3</p>
      </div>
      <div class="card red">
        <h3>3. Fine-Tuned LLM (QLoRA)</h3>
        <p>4-bit NF4 квантизация, LoRA r=16.<br>31,850 примера, 2 епохи, A100 GPU.<br>Тестван: Qwen3-8B (Ministral 8B \u2014 FP8 проблем)</p>
      </div>
      <div class="card green">
        <h3>4. Classic ML (TF-IDF + XGBoost)</h3>
        <p>50K TF-IDF признака, 13 бинарни класификатора.<br>300 дървета, depth 6, scale_pos_weight.<br>Без GPU \u2014 бърз, евтин, стабилен.</p>
      </div>
    </div>
    <p class="footnote">Всички модели оценени на един и същ замразен тест сет (8,044 доклада)</p>
  </div>
</section>

<!-- ═══ SLIDE 6: CLASSIC ML PIPELINE (NEW) ═══ -->
<section class="slide light" data-slide="5">
  <div class="header-bar"><h2>Classic ML \u2014 TF-IDF + XGBoost Pipeline</h2></div>
  <div class="content">
    <div class="pipeline">
      <div class="pipe-step" style="border-top-color:var(--steel);"><h4>Наратив</h4><p>Текст от пилот<br>(~200 думи)</p></div>
      <span class="pipe-arrow">\u2192</span>
      <div class="pipe-step" style="border-top-color:var(--blue);"><h4>TF-IDF</h4><p>50K признака<br>ngram(1,2)</p></div>
      <span class="pipe-arrow">\u2192</span>
      <div class="pipe-step" style="border-top-color:var(--sky);"><h4>Sparse Matrix</h4><p>31,850 \u00d7 50,000<br>sublinear TF</p></div>
      <span class="pipe-arrow">\u2192</span>
      <div class="pipe-step" style="border-top-color:var(--orange);"><h4>13\u00d7 XGBoost</h4><p>Бинарен<br>класификатор</p></div>
      <span class="pipe-arrow">\u2192</span>
      <div class="pipe-step" style="border-top-color:var(--green);"><h4>Предикции</h4><p>13 вероятности<br>(multi-label)</p></div>
    </div>
    <div class="three-col" style="flex:1;">
      <div class="diagram-box accent-blue">
        <h4>TF-IDF Vectorizer</h4>
        <p>Term Frequency \u2013 Inverse Document Frequency. Претегля думи по важност в документа спрямо целия корпус. Улавя domain-специфични n-грами: "runway incursion", "TCAS RA".</p>
      </div>
      <div class="diagram-box accent-orange">
        <h4>XGBoost (One-vs-Rest)</h4>
        <p>13 независими бинарни класификатора. Всеки има собствен scale_pos_weight за справяне с дисбаланса. 300 дървета, depth 6, lr 0.1.</p>
      </div>
      <div class="diagram-box accent-green">
        <h4>Защо работи?</h4>
        <p>Per-label оптимизация: всяка категория има собствен модел и праг. Градирани вероятности (0\u20131). Бърз: &lt;1 мин inference без GPU.</p>
      </div>
    </div>
    <p class="footnote">Hyperparameter tuning (3-fold CV, 8 TF-IDF + 3 модела) потвърждава: baseline е оптимален (\u0394 Macro-F1 &lt; 0.005)</p>
  </div>
</section>

<!-- ═══ SLIDE 7: LLM ARCHITECTURE (NEW) ═══ -->
<section class="slide light" data-slide="6">
  <div class="header-bar"><h2>LLM архитектура \u2014 Dense vs MoE</h2></div>
  <div class="content">
    <div class="two-col" style="gap:24px;">
      <div class="col" style="gap:16px;">
        <div class="diagram-box accent-blue">
          <h4>Dense Transformer</h4>
          <p>Всички параметри активни при inference.<br>Qwen3-8B: 8B параметра, всички активни.<br>По-малък, по-бърз, по-евтин. Лесен за fine-tuning (QLoRA).</p>
        </div>
        <div class="diagram-box accent-orange">
          <h4>Mixture of Experts (MoE)</h4>
          <p>Маршрутизатор избира подмножество експерти.<br>Mistral Large 3: 675B (41B активни).<br>DeepSeek V3.2: 671B MoE. По-мощен, изисква API.</p>
        </div>
      </div>
      <div class="col">
        <table>
          <thead><tr><th>Модел</th><th>Архитектура</th><th>Параметри</th><th>Лиценз</th></tr></thead>
          <tbody>
            <tr><td>Qwen3-8B</td><td>Dense</td><td>8B</td><td>Apache 2.0</td></tr>
            <tr><td>Mistral Large 3</td><td>MoE</td><td>675B (41B act.)</td><td>Apache 2.0</td></tr>
            <tr><td>DeepSeek V3.2</td><td>MoE</td><td>671B</td><td>MIT</td></tr>
            <tr><td>Ministral 8B</td><td>Dense (FP8)</td><td>8B</td><td>Apache 2.0</td></tr>
          </tbody>
        </table>
        <div class="diagram-box" style="margin-top:12px;">
          <h4>Инфраструктура</h4>
          <p>Qwen3-8B: Modal GPU (L4 inference, A100 training) + vLLM<br>
          Mistral Large 3: Batch API (безплатен план, ~5 мин)<br>
          DeepSeek V3.2: DeepInfra API (prefix caching 62\u201382%)</p>
        </div>
      </div>
    </div>
    <p class="footnote">Общо: 22 експеримента на 4 модела (3 LLM + 1 Classic ML)</p>
  </div>
</section>

<!-- ═══ SLIDE 8: QLORA FINE-TUNING (NEW) ═══ -->
<section class="slide light" data-slide="7">
  <div class="header-bar"><h2>QLoRA Fine-Tuning</h2></div>
  <div class="content">
    <div class="two-col" style="gap:24px;">
      <div class="col" style="align-items:center; justify-content:center;">
        <div style="border:2px dashed var(--muted); border-radius:12px; padding:24px; text-align:center; background:var(--light-gray); width:100%; max-width:380px;">
          <p style="font-family:Georgia,serif; font-size:16px; color:var(--muted);">Qwen3-8B<br>(замразен, 4-bit NF4)</p>
          <div style="display:flex; gap:16px; justify-content:center; margin-top:16px;">
            <div style="border:2px solid var(--red); border-radius:8px; padding:12px 20px; background:white;">
              <p style="font-family:Consolas,monospace; font-size:13px; color:var(--red); font-weight:700;">LoRA A<br>(q_proj)</p>
            </div>
            <div style="border:2px solid var(--red); border-radius:8px; padding:12px 20px; background:white;">
              <p style="font-family:Consolas,monospace; font-size:13px; color:var(--red); font-weight:700;">LoRA B<br>(v_proj)</p>
            </div>
          </div>
          <p style="font-family:Consolas,monospace; font-size:11px; color:var(--text); margin-top:12px;">r=16, \u03b1=16, dropout=0.05</p>
          <p style="color:var(--red); font-weight:700; font-size:13px; margin-top:8px;">Trainable: 0.2% от параметрите</p>
        </div>
      </div>
      <div class="col">
        <table class="mini-table">
          <thead><tr><th>Параметър</th><th>Стойност</th></tr></thead>
          <tbody>
            <tr><td>Квантизация</td><td>4-bit NF4 (double quant)</td></tr>
            <tr><td>LoRA rank</td><td>r=16, \u03b1=16</td></tr>
            <tr><td>Target modules</td><td>q_proj, v_proj</td></tr>
            <tr><td>Training data</td><td>31,850 примера</td></tr>
            <tr><td>Epochs</td><td>2 (3,982 стъпки)</td></tr>
            <tr><td>Batch size</td><td>4 (grad accum \u00d74)</td></tr>
            <tr><td>Learning rate</td><td>2e-5 (cosine)</td></tr>
            <tr><td>Optimizer</td><td>paged_adamw_8bit</td></tr>
            <tr><td>GPU</td><td>A100 80GB (Modal)</td></tr>
            <tr><td>Времетраене</td><td>3h 47min</td></tr>
            <tr><td>Финална загуба</td><td>1.691 (66.8% acc)</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div class="warning-box" style="flex-shrink:0;">
      \u26a0 Ministral 8B: Mistral3ForConditionalGeneration (мултимодален) с FP8 \u2014 не позволява QLoRA (4-bit NF4). Fine-tuning не подобри резултатите (Macro-F1: 0.489 vs 0.491 zero-shot). Заменен с Qwen3-8B.
    </div>
  </div>
</section>

<!-- ═══ SLIDE 9: PROMPT ENGINEERING (NEW) ═══ -->
<section class="slide light" data-slide="8">
  <div class="header-bar"><h2>Prompt Engineering \u2014 Taxonomy-Enriched</h2></div>
  <div class="content">
    <div class="vs-compare">
      <div class="vs-box">
        <p class="code-label">Basic Prompt</p>
        <div class="code-block">System: Classify this ASRS report.
Categories:
- Aircraft Equipment Problem
- Airspace Violation
- ATC Issue
- Conflict
- Deviation - Altitude
- Deviation - Procedural
- Deviation - Speed
- Deviation - Track/Heading
- Flight Deck/Cabin Event
- Ground Event/Encounter
- Ground Excursion
- Ground Incursion
- Inflight Event/Encounter

Return JSON list of matching categories.</div>
      </div>
      <div class="vs-box">
        <p class="code-label">Taxonomy-Enriched Prompt</p>
        <div class="code-block">System: Classify using NASA ASRS taxonomy.
Categories with subcategories:

- Aircraft Equipment Problem
  Less Severe | Critical
  Hint: mechanical failures, malfunctions

- Conflict
  NMAC | Airborne | Ground Conflict
  Hint: TCAS RA, traffic proximity

- Deviation - Altitude
  Overshoot | Undershoot | Crossing
  Hint: assigned vs actual altitude
...
Return JSON list of matching categories.</div>
      </div>
    </div>
    <div class="stat-row" style="flex-shrink:0; margin-top:12px;">
      <div class="stat-card" style="flex:1; border-top:4px solid var(--green);">
        <div class="value" style="color:var(--green); font-size:28px;">+0.040</div>
        <div class="label">Macro-F1 подобрение (Qwen3-8B ZS)</div>
      </div>
      <div class="stat-card" style="flex:1; border-top:4px solid var(--sky);">
        <div class="value" style="color:var(--sky); font-size:28px;">+0.133</div>
        <div class="label">Micro-F1 подобрение (Qwen3-8B ZS)</div>
      </div>
      <div class="stat-card" style="flex:1; border-top:4px solid var(--orange);">
        <div class="value" style="color:var(--orange); font-size:28px;">+0.073</div>
        <div class="label">Macro-F1 подобрение (Qwen3-8B FS)</div>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 10: CLASSIC ML RESULTS (NEW) ═══ -->
<section class="slide light" data-slide="9">
  <div class="header-bar"><h2>Резултати \u2014 Classic ML (TF-IDF + XGBoost)</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main">
        ${images.classicBar ? `<img src="${images.classicBar}" alt="Classic ML F1 bar chart">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <div class="stat-card" style="border-top:4px solid var(--green);">
          <div class="value" style="color:var(--green); font-size:28px;">0.691</div>
          <div class="label">Macro-F1</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--blue);">
          <div class="value" style="color:var(--blue); font-size:28px;">0.746</div>
          <div class="label">Micro-F1</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--sky);">
          <div class="value" style="color:var(--sky); font-size:22px;">AUC: 0.932</div>
          <div class="label">Macro-AUC</div>
        </div>
        <div style="font-size:12px;">
          <p style="font-weight:700; margin-bottom:4px; color:var(--green);">Най-добри:</p>
          <p style="color:var(--muted);">\u2022 Aircraft Equipment: 0.816<br>\u2022 Conflict: 0.801<br>\u2022 Deviation-Procedural: 0.795</p>
          <p style="font-weight:700; margin:8px 0 4px; color:var(--red);">Най-слаби:</p>
          <p style="color:var(--muted);">\u2022 Airspace Violation: 0.568<br>\u2022 Ground Excursion: 0.572<br>\u2022 Deviation-Speed: 0.577</p>
        </div>
      </div>
    </div>
    <p class="footnote">Редките категории са най-трудни \u2014 дисбаланс 30.3\u00d7 между най-честата и най-рядката</p>
  </div>
</section>

<!-- ═══ SLIDE 11: ZERO-SHOT RESULTS (NEW) ═══ -->
<section class="slide light" data-slide="10">
  <div class="header-bar"><h2>Резултати \u2014 Zero-Shot LLM</h2></div>
  <div class="content">
    <table>
      <thead><tr><th>Модел</th><th>Промпт</th><th>Macro-F1</th><th>Micro-F1</th><th>AUC</th></tr></thead>
      <tbody>
        <tr><td>DeepSeek V3.2 + thinking</td><td>taxonomy</td><td><strong>0.681</strong></td><td><strong>0.723</strong></td><td><strong>0.810</strong></td></tr>
        <tr><td>Mistral Large 3</td><td>taxonomy</td><td>0.658</td><td>0.712</td><td>0.793</td></tr>
        <tr><td>DeepSeek V3.2</td><td>taxonomy</td><td>0.623</td><td>0.693</td><td>0.746</td></tr>
        <tr><td>Qwen3-8B</td><td>taxonomy</td><td>0.499</td><td>0.605</td><td>0.701</td></tr>
        <tr><td>Ministral 8B</td><td>basic</td><td>0.491</td><td>0.543</td><td>0.744</td></tr>
        <tr><td>Qwen3-8B</td><td>basic</td><td>0.459</td><td>0.473</td><td>0.727</td></tr>
      </tbody>
    </table>
    <div class="three-col" style="margin-top:12px; flex-shrink:0;">
      <div class="diagram-box accent-sky">
        <h4>Мащаб е решаващ</h4>
        <p>671\u2013675B MoE (0.658\u20130.681) >> 8B Dense (0.459\u20130.499). Разлика: +0.16\u20130.22 Macro-F1.</p>
      </div>
      <div class="diagram-box accent-orange">
        <h4>Taxonomy помага</h4>
        <p>Qwen3-8B: +0.040 Macro-F1, +0.133 Micro-F1 с taxonomy. По-ясни категорийни граници.</p>
      </div>
      <div class="diagram-box accent-red">
        <h4>Thinking: +0.058 при 671B</h4>
        <p>DeepSeek: 0.681 vs 0.623. Но 45\u00d7 по-бавно и 4.8\u00d7 по-скъпо. При 8B: +0.007.</p>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 12: FEW-SHOT + FINE-TUNING (NEW) ═══ -->
<section class="slide light" data-slide="11">
  <div class="header-bar"><h2>Резултати \u2014 Few-Shot и Fine-Tuning</h2></div>
  <div class="content">
    <div class="vs-compare">
      <div class="vs-box">
        <h3 style="color:var(--orange); margin-bottom:8px;">Few-Shot LLM</h3>
        <table class="mini-table">
          <thead><tr><th>Модел</th><th>Промпт</th><th>Macro-F1</th><th>Micro-F1</th></tr></thead>
          <tbody>
            <tr><td>Mistral Large 3</td><td>taxonomy</td><td><strong>0.640</strong></td><td><strong>0.686</strong></td></tr>
            <tr><td>Ministral 8B</td><td>basic</td><td>0.540</td><td>0.536</td></tr>
            <tr><td>Qwen3-8B + thinking</td><td>taxonomy</td><td>0.533</td><td>0.556</td></tr>
            <tr><td>Qwen3-8B</td><td>taxonomy</td><td>0.526</td><td>0.544</td></tr>
            <tr><td>Qwen3-8B</td><td>basic</td><td>0.453</td><td>0.468</td></tr>
          </tbody>
        </table>
      </div>
      <div class="vs-box">
        <h3 style="color:var(--red); margin-bottom:8px;">Fine-Tuned LLM (QLoRA)</h3>
        <table class="mini-table">
          <thead><tr><th>Модел</th><th>Macro-F1</th><th>Micro-F1</th><th>AUC</th></tr></thead>
          <tbody>
            <tr><td>Qwen3-8B QLoRA</td><td><strong>0.510</strong></td><td><strong>0.632</strong></td><td>0.700</td></tr>
            <tr><td>Ministral 8B LoRA/FP8</td><td>0.489</td><td>0.542</td><td>0.744</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div style="display:flex; flex-direction:column; gap:6px; flex-shrink:0; margin-top:8px;">
      <div class="finding" style="--c:var(--orange);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--orange);"></div>
        <h3 style="color:var(--orange);">Few-shot: мащабът доминира</h3>
        <p>Mistral Large 3 (675B) > всички 8B. Few-shot на малък модел е по-слаб от zero-shot на голям.</p>
      </div>
      <div class="finding" style="--c:var(--red);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--red);"></div>
        <h3 style="color:var(--red);">Fine-tuning: +0.159 Micro-F1</h3>
        <p>Qwen3-8B QLoRA: значимо подобрение на Micro-F1, но Macro-F1 остава по-нисък от ZS на 675B модел.</p>
      </div>
      <div class="finding" style="--c:var(--sky);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--sky);"></div>
        <h3 style="color:var(--sky);">Thinking mode: минимален ефект при 8B</h3>
        <p>FS taxonomy + thinking: +0.007 Macro-F1 vs no-thinking. Не оправдава 6\u00d7 по-висока цена (A100 vs L4).</p>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 13: GRAND COMPARISON ═══ -->
<section class="slide light" data-slide="12">
  <div class="header-bar"><h2>Сравнение на всички модели (13 категории)</h2></div>
  <div class="content">
    <div class="fig-container" style="flex:2;">
      ${images.grand ? `<img src="${images.grand}" alt="Grand comparison">` : '<p style="color:var(--muted);">[Image not found]</p>'}
    </div>
    <table style="flex-shrink:0;">
      <thead><tr><th>Подход</th><th>Най-добър модел</th><th>Macro-F1</th><th>Micro-F1</th></tr></thead>
      <tbody>
        <tr><td>Classic ML</td><td>TF-IDF + XGBoost</td><td><strong>0.691</strong></td><td><strong>0.746</strong></td></tr>
        <tr><td>Zero-Shot</td><td>DeepSeek V3.2 + thinking</td><td>0.681</td><td>0.723</td></tr>
        <tr><td>Few-Shot</td><td>Mistral Large 3</td><td>0.640</td><td>0.686</td></tr>
        <tr><td>Fine-Tuned</td><td>Qwen3-8B QLoRA</td><td>0.510</td><td>0.632</td></tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ═══ SLIDE 14: CATEGORY HEATMAP ═══ -->
<section class="slide light" data-slide="13">
  <div class="header-bar"><h2>F1 по категория и модел (хийтмап)</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main" style="flex:3;">
        ${images.categoryHeat ? `<img src="${images.categoryHeat}" alt="Category heatmap">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <div class="stat-card" style="border-top:4px solid var(--green);">
          <div class="value" style="color:var(--green); font-size:28px;">7 / 13</div>
          <div class="label">категории \u2014<br>XGBoost най-добър</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--sky);">
          <div class="value" style="color:var(--sky); font-size:28px;">6 / 13</div>
          <div class="label">категории \u2014<br>DeepSeek V3.2 + thinking</div>
        </div>
        <div style="font-size:12px; padding:8px;">
          <p style="font-weight:700; margin-bottom:6px;">Семантичен анализ:</p>
          <p style="color:var(--muted); line-height:1.4;">XGBoost доминира при категории с ясни текстови маркери (ATC Issue, Ground Incursion).<br><br>LLM печелят при категории изискващи разбиране на контекста (Conflict, Deviation-Altitude).</p>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 15: SCALE EFFECT ═══ -->
<section class="slide light" data-slide="14">
  <div class="header-bar"><h2>Йерархия на факторите за LLM представяне</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main">
        ${images.scale ? `<img src="${images.scale}" alt="Scale effect">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <div class="hierarchy">
          <div class="hierarchy-item" style="border-left:3px solid var(--sky);">
            <div class="num" style="background:var(--sky);">1</div>
            <h4 style="color:var(--sky);">Мащаб на модела</h4>
            <p>675B \u2192 0.658\u20130.681 | 8B \u2192 0.459\u20130.510<br>\u0394: +0.16\u20130.22 Macro-F1</p>
          </div>
          <div class="hierarchy-item" style="border-left:3px solid var(--orange);">
            <div class="num" style="background:var(--orange);">2</div>
            <h4 style="color:var(--orange);">Prompt Engineering</h4>
            <p>Taxonomy: +0.040\u20130.073<br>Few-shot: +0.067 (Mistral Large 3)</p>
          </div>
          <div class="hierarchy-item" style="border-left:3px solid var(--red);">
            <div class="num" style="background:var(--red);">3</div>
            <h4 style="color:var(--red);">Fine-Tuning</h4>
            <p>QLoRA: +0.051 Macro-F1<br>+0.159 Micro-F1 (най-добър за 8B)</p>
          </div>
          <div class="hierarchy-item" style="border-left:3px solid var(--muted);">
            <div class="num" style="background:var(--muted);">4</div>
            <h4 style="color:var(--muted);">Thinking Mode</h4>
            <p>671B: +0.058 | 8B: +0.007<br>48 лейбъла: \u22120.003 (вреди)</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 16: SUBCATEGORY RESULTS (NEW) ═══ -->
<section class="slide light" data-slide="15">
  <div class="header-bar"><h2>Подкатегории \u2014 48-лейбъл класификация</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main">
        ${images.parentVsSub ? `<img src="${images.parentVsSub}" alt="Parent vs subcategory">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <table class="mini-table">
          <thead><tr><th>Модел</th><th>Parent</th><th>Subcat</th><th>\u0394</th></tr></thead>
          <tbody>
            <tr><td>Classic ML</td><td>0.691</td><td>0.510</td><td>\u22120.181</td></tr>
            <tr><td>Mistral Large 3</td><td>0.658</td><td>0.449</td><td>\u22120.209</td></tr>
            <tr><td>DeepSeek V3.2</td><td>0.623</td><td>0.422</td><td>\u22120.201</td></tr>
            <tr><td>Qwen3-8B</td><td>0.499</td><td>0.235</td><td>\u22120.264</td></tr>
          </tbody>
        </table>
        <div style="font-size:11px; padding:4px;">
          <p style="font-weight:700; margin-bottom:4px;">Ключови наблюдения:</p>
          <p style="color:var(--muted); line-height:1.4;">
            \u2022 Classic ML запазва AUC: 0.934 vs 0.932<br>
            \u2022 Най-трудни: Ground Event (F1=0.000\u20130.164)<br>
            \u2022 Най-добри: Haz Mat (0.824), Smoke/Fire (0.815)<br>
            \u2022 LLM деградация > Classic ML (\u22120.264 vs \u22120.181)<br>
            \u2022 Mistral Large 3 бие ML на 11/48 подкатегории
          </p>
        </div>
      </div>
    </div>
    <p class="footnote">48-лейбъл задачата е значително по-трудна: Macro-F1 пада с 0.18\u20130.26 за всички модели</p>
  </div>
</section>

<!-- ═══ SLIDE 17: COST VS PERFORMANCE ═══ -->
<section class="slide light" data-slide="16">
  <div class="header-bar"><h2>Цена срещу представяне</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main" style="flex:2;">
        ${images.cost ? `<img src="${images.cost}" alt="Cost vs performance">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <table class="mini-table">
          <thead><tr><th>Подход</th><th>Цена</th><th>Време</th><th>Macro-F1</th></tr></thead>
          <tbody>
            <tr><td>Classic ML</td><td>$0</td><td>&lt;1 min</td><td>0.691</td></tr>
            <tr><td>Mistral Large 3</td><td>$0</td><td>~5 min</td><td>0.658</td></tr>
            <tr><td>DeepSeek V3.2</td><td>$1.39</td><td>6.5 min</td><td>0.623</td></tr>
            <tr><td>DS V3.2 + thinking</td><td>$6.73</td><td>~5 hr</td><td>0.681</td></tr>
            <tr><td>Qwen3-8B (all)</td><td>~$19</td><td>various</td><td>0.510</td></tr>
          </tbody>
        </table>
        <div class="stat-card" style="border-top:4px solid var(--sky);">
          <div class="value" style="color:var(--sky); font-size:24px;">~$53</div>
          <div class="label">Обща цена на 22 експеримента<br>(Modal: $38 + DeepInfra: $15)</div>
        </div>
      </div>
    </div>
    <p class="footnote">Classic ML е едновременно най-евтин, най-бърз, и най-точен за 13-лейбъл задачата</p>
  </div>
</section>

<!-- ═══ SLIDE 18: WHY CLASSIC ML WINS (NEW) ═══ -->
<section class="slide light" data-slide="17">
  <div class="header-bar"><h2>Защо Classic ML печели?</h2></div>
  <div class="content">
    <div class="reason-grid">
      <div class="card green">
        <h3>Domain-Specific N-грами</h3>
        <p>TF-IDF улавя авиационни термини: "runway incursion", "TCAS RA", "altitude deviation". Тези биграми директно кореспондират с имената на категориите \u2014 силен сигнал.</p>
      </div>
      <div class="card blue">
        <h3>Per-Label оптимизация</h3>
        <p>13 независими класификатора, всеки с собствен праг и scale_pos_weight. LLM правят една обща предикция за всички категории наведнъж.</p>
      </div>
      <div class="card orange">
        <h3>Достатъчно данни за BoW</h3>
        <p>31,850 тренировъчни примера са достатъчни за bag-of-words подход. LLM от 8B нямат достатъчно капацитет, а 675B не могат да се fine-tune (само API).</p>
      </div>
      <div class="card red">
        <h3>Градирани вероятности</h3>
        <p>XGBoost дава вероятности (0\u20131) за всяка категория. LLM дават бинарно да/не. Оптималният праг може да се настрои. AUC: 0.932 vs 0.810.</p>
      </div>
    </div>
    <p class="footnote">Класическото ML не е "остаряло" \u2014 за структурирани задачи с ясни текстови маркери, то остава оптималният избор</p>
  </div>
</section>

<!-- ═══ SLIDE 19: KEY FINDINGS ═══ -->
<section class="slide light" data-slide="18">
  <div class="header-bar"><h2>Ключови находки</h2></div>
  <div class="content" style="gap:4px;">
    <div class="finding"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--green);"></div>
      <h3 style="color:var(--green);">Classic ML доминира</h3>
      <p>TF-IDF + XGBoost (Macro-F1 0.691) превъзхожда всички LLM. Без GPU, &lt;1 мин inference, $0 цена.</p>
    </div>
    <div class="finding"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--sky);"></div>
      <h3 style="color:var(--sky);">Размерът на модела е решаващ</h3>
      <p>675B MoE (0.658\u20130.681) >> 8B Dense (0.459\u20130.510). Fine-tuning на 8B не компенсира разликата.</p>
    </div>
    <div class="finding"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--orange);"></div>
      <h3 style="color:var(--orange);">Taxonomy промптинг помага</h3>
      <p>Подкатегории + подсказки: +0.040 Macro-F1, +0.133 Micro-F1 (Qwen3-8B ZS). Особено за малки модели.</p>
    </div>
    <div class="finding"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--red);"></div>
      <h3 style="color:var(--red);">Thinking mode \u2014 нюансиран ефект</h3>
      <p>671B: +0.058 за 13 лейбъла, но \u22120.003 за 48 лейбъла (21.6% parse failures). 8B: +0.007.</p>
    </div>
    <div class="finding"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--steel);"></div>
      <h3 style="color:var(--steel);">48-лейбъл задачата е значително по-трудна</h3>
      <p>Macro-F1 пада с 0.18\u20130.26 за всички модели. Classic ML запазва AUC (0.934), LLM деградират повече.</p>
    </div>
    <div class="recommendation-box">
      \u2705 &nbsp; Препоръка: За production класификация на ASRS доклади \u2014 TF-IDF + XGBoost с per-label оптимизация. LLM са полезни за exploration и когато няма обучителни данни.
    </div>
  </div>
</section>

<!-- ═══ SLIDE 20: CONCLUSION ═══ -->
<section class="slide dark" data-slide="19">
  <div class="content centered">
    <h1 style="font-size:30px;">Заключение</h1>
    <hr style="border:none; border-top:2px solid var(--sky); width:300px; margin:16px auto;">
    <p style="font-size:17px; line-height:1.5; max-width:800px; margin:0 auto;">
      За мултилейбълна класификация на авиационни доклади,<br>
      класическото ML (TF-IDF + XGBoost) остава най-ефективният подход \u2014<br>
      най-висок F1, нулева цена за inference, и най-бърз.
    </p>
    <h3 style="color:var(--sky); margin-top:20px; font-size:19px;">Приноси</h3>
    <div style="text-align:left; display:inline-block; margin:8px auto;">
      <p style="font-size:14px; margin:6px 0;">1. Систематично сравнение на 4 подхода в 22 експеримента</p>
      <p style="font-size:14px; margin:6px 0;">2. Taxonomy-enriched prompting методология (+0.040\u20130.073 Macro-F1)</p>
      <p style="font-size:14px; margin:6px 0;">3. Пълен open-source codebase с възпроизводими резултати ($53)</p>
    </div>
    <h3 style="color:var(--sky); margin-top:16px; font-size:19px;">Бъдещи насоки</h3>
    <ul class="future-list">
      <li>Encoder-based модели (BERT, DeBERTa) за класификация</li>
      <li>RAG подход с вградени ASRS таксономии</li>
      <li>Мултимодално обучение (текст + структурирани полета)</li>
      <li>Поточна класификация за реално време</li>
    </ul>
    <hr style="border:none; border-top:2px solid var(--sky); width:300px; margin:16px auto 8px;">
    <div style="display:flex; justify-content:center; gap:40px; align-items:center;">
      <p style="font-family:Georgia,serif; font-size:21px; color:var(--sky); font-weight:700;">Благодаря за вниманието!</p>
      <p style="font-family:Consolas,monospace; font-size:12px; color:var(--muted);">github.com/rashevzarko-crypto/NBU-ASRS</p>
    </div>
    <p style="color:var(--muted); font-size:12px; margin-top:6px;">Зарко Рашев &nbsp;|&nbsp; Ф№ F98363 &nbsp;|&nbsp; НБУ &nbsp;|&nbsp; 2026 &nbsp;|&nbsp; Бюджет: ~$53</p>
  </div>
</section>

<!-- ═══ CONTROLS ═══ -->
<div class="slide-counter" id="counter">1 / 20</div>
<div class="progress-bar" id="progress" style="width:5%;"></div>

<script>
(function() {
  const slides = document.querySelectorAll('.slide');
  const counter = document.getElementById('counter');
  const progress = document.getElementById('progress');
  const total = slides.length;
  let current = 0;
  let overviewMode = false;

  function goTo(n) {
    if (n < 0 || n >= total) return;
    slides[current].classList.remove('active');
    current = n;
    slides[current].classList.add('active');
    counter.textContent = (current + 1) + ' / ' + total;
    progress.style.width = ((current + 1) / total * 100).toFixed(2) + '%';
    if (slides[current].classList.contains('dark')) {
      counter.style.color = 'rgba(255,255,255,0.4)';
    } else {
      counter.style.color = 'rgba(0,0,0,0.3)';
    }
  }

  function toggleOverview() {
    overviewMode = !overviewMode;
    document.body.classList.toggle('overview', overviewMode);
    if (overviewMode) {
      slides.forEach(function(s) { s.style.opacity = '1'; s.style.pointerEvents = 'auto'; });
    } else {
      slides.forEach(function(s, i) {
        s.style.opacity = i === current ? '1' : '0';
        s.style.pointerEvents = i === current ? 'auto' : 'none';
      });
    }
  }

  document.addEventListener('keydown', function(e) {
    if (overviewMode && e.key === 'Escape') { toggleOverview(); return; }
    if (e.key === 'Escape') { toggleOverview(); return; }
    if (overviewMode) return;
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); goTo(current + 1); }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); goTo(current - 1); }
    if (e.key === 'Home') { e.preventDefault(); goTo(0); }
    if (e.key === 'End') { e.preventDefault(); goTo(total - 1); }
  });

  document.addEventListener('click', function(e) {
    if (overviewMode) {
      var el = e.target;
      while (el && !el.classList.contains('slide')) el = el.parentElement;
      if (el) {
        var idx = parseInt(el.getAttribute('data-slide'));
        toggleOverview();
        goTo(idx);
      }
      return;
    }
    if (e.clientX < window.innerWidth / 2) goTo(current - 1);
    else goTo(current + 1);
  });

  var touchStart = 0;
  document.addEventListener('touchstart', function(e) { touchStart = e.changedTouches[0].clientX; });
  document.addEventListener('touchend', function(e) {
    var diff = e.changedTouches[0].clientX - touchStart;
    if (Math.abs(diff) > 50) {
      if (diff < 0) goTo(current + 1); else goTo(current - 1);
    }
  });
})();
</script>
</body>
</html>`;

  const outPath = path.join(ROOT, "defense_presentation.html");
  fs.writeFileSync(outPath, html, "utf-8");
  console.log(`Written: ${outPath}`);
  console.log(`File size: ${(fs.statSync(outPath).size / 1024 / 1024).toFixed(1)} MB`);
  console.log(`Total slides: 20`);
}

main();
