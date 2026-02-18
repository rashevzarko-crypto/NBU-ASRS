/**
 * NBU Thesis Defense Presentation Generator — HTML
 * Generates defense_presentation.html: self-contained, keyboard-navigable, base64 images.
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
  console.log("Generating defense_presentation.html...");

  const images = {
    heatmap:        loadBase64("co_occurrence_heatmap.png"),
    grand:          loadBase64("fig_grand_comparison.png"),
    approach:       loadBase64("fig_approach_summary.png"),
    categoryHeat:   loadBase64("fig_category_heatmap.png"),
    cost:           loadBase64("fig_cost_vs_performance.png"),
    scale:          loadBase64("fig_scale_effect.png"),
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
  .card h3 { font-size: 17px; margin-bottom: 8px; }
  .card p { font-size: 14px; color: var(--muted); line-height: 1.5; }

  /* ─── FINDING CARDS ─── */
  .finding {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 12px 16px 12px 22px; position: relative; margin: 6px 0;
  }
  .finding::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 5px; border-radius: 8px 0 0 8px;
  }
  .finding h3 { font-size: 15px; margin-bottom: 4px; }
  .finding p { font-size: 12.5px; color: var(--muted); line-height: 1.4; }

  /* ─── TABLE ─── */
  table { width: 100%; border-collapse: collapse; margin: 16px 0; }
  thead th {
    background: var(--steel); color: var(--white);
    padding: 10px 12px; font-size: 14px; text-align: center;
  }
  tbody td {
    padding: 8px 12px; text-align: center; font-size: 13px; border-bottom: 1px solid var(--light-gray);
  }
  tbody tr:nth-child(even) { background: var(--light-gray); }
  tbody td:first-child { font-family: Georgia, serif; font-weight: 600; text-align: left; }

  /* ─── BEFORE/AFTER ─── */
  .before-after { display: flex; gap: 40px; align-items: center; justify-content: center; flex: 1; }
  .ba-box {
    background: var(--white); border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 24px 32px; text-align: center; width: 320px;
  }
  .ba-box .big-num { font-family: Consolas, monospace; font-size: 56px; font-weight: 700; margin: 8px 0; }
  .ba-box .ba-label { font-size: 14px; color: var(--muted); }
  .ba-box .ba-detail { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .ba-arrow { font-size: 48px; color: var(--sky); }
  .lesson-box {
    background: #FFF3CD; border-radius: 6px; padding: 14px 20px;
    font-size: 14px; color: var(--text); text-align: center; margin-top: 12px;
  }

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

  /* ─── CONTROLS ─── */
  .slide-counter {
    position: fixed; bottom: 16px; right: 24px;
    font-family: Consolas, monospace; font-size: 14px; color: rgba(255,255,255,0.5);
    z-index: 100; pointer-events: none;
  }
  .light .slide-counter, .progress-bar-container ~ .slide-counter { color: rgba(0,0,0,0.3); }
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

  /* ─── DARK SLIDE TEXT ─── */
  .dark .stat-card .label { color: var(--muted); }

  /* ─── RESPONSIVE ─── */
  @media (max-width: 900px) {
    .two-col { flex-direction: column; }
    .card-grid { grid-template-columns: 1fr; }
    .before-after { flex-direction: column; gap: 16px; }
    .fig-with-side { flex-direction: column; }
  }

  /* ─── PRINT ─── */
  @media print {
    .slide { position: relative !important; opacity: 1 !important; pointer-events: auto !important;
      page-break-after: always; width: 100vw; height: 100vh; }
    .progress-bar, .slide-counter { display: none; }
  }

  /* ─── SUB QUESTIONS ─── */
  .sub-questions { list-style: none; padding: 0; margin-top: 12px; }
  .sub-questions li { font-size: 14px; margin: 8px 0; padding-left: 8px; line-height: 1.4; }
  .sub-questions li::before { content: none; }

  /* ─── STAT ROW ─── */
  .stat-row { display: flex; gap: 12px; }

  /* ─── KEY STATS ─── */
  .key-stats { display: flex; flex-direction: column; gap: 12px; }
  .key-stat { display: flex; align-items: baseline; gap: 12px; }
  .key-stat .ks-val { font-family: Consolas, monospace; font-size: 20px; font-weight: 700; color: var(--sky); min-width: 120px; text-align: right; }
  .key-stat .ks-label { font-size: 13px; color: var(--text); }

  /* ─── METRICS ROW ─── */
  .metrics-row { display: flex; gap: 12px; justify-content: center; margin-top: 8px; flex-shrink: 0; }
  .metric-card {
    background: var(--white); border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    padding: 10px 16px; text-align: center; flex: 1; max-width: 240px;
  }
  .metric-card .mc-approach { font-family: Georgia, serif; font-size: 12px; font-weight: 700; }
  .metric-card .mc-detail { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* ─── FUTURE WORK ─── */
  .future-list { list-style: disc; padding-left: 24px; text-align: left; display: inline-block; }
  .future-list li { font-size: 16px; margin: 8px 0; color: var(--white); }
</style>
</head>
<body>

<!-- ═══ SLIDE 1: TITLE ═══ -->
<section class="slide dark active" data-slide="0">
  <div class="content centered">
    <p style="color:var(--sky); font-style:italic; font-size:17px; margin-bottom:16px;">Нов български университет</p>
    <h1 style="font-size:36px; line-height:1.3; max-width:900px;">Мултимодално дълбоко обучение за<br>класификация на инциденти по ВПП</h1>
    <p class="subtitle" style="font-size:19px; margin-top:16px;">Multi-Label Classification of Aviation Safety Reports<br>Using LLMs and Classic ML</p>
    <hr style="border:none; border-top:2px solid var(--sky); width:300px; margin:24px auto;">
    <p style="color:var(--muted); font-size:16px;">Жарко Рашев &nbsp;|&nbsp; Магистърска теза &nbsp;|&nbsp; 2026</p>
  </div>
</section>

<!-- ═══ SLIDE 2: RESEARCH QUESTION ═══ -->
<section class="slide light" data-slide="1">
  <div class="header-bar"><h2>Изследователски въпрос</h2></div>
  <div class="content">
    <div class="two-col">
      <div class="col">
        <p style="font-family:Georgia,serif; font-size:19px; font-style:italic; line-height:1.5; margin-bottom:20px;">
          Могат ли големите езикови модели (LLM) да конкурират класическото машинно обучение при мултилейбълна класификация на авиационни доклади за безопасност?
        </p>
        <ol class="sub-questions">
          <li>1. Как се сравняват zero-shot, few-shot и fine-tuned LLM подходи?</li>
          <li>2. Каква е ролята на размера на модела (8B vs 675B)?</li>
          <li>3. Кога taxonomy-enriched промптинг помага?</li>
          <li>4. Какъв е компромисът между цена и качество?</li>
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
            <div class="label">категории аномалии</div>
          </div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--orange);">
          <div class="value" style="color:var(--orange);">78%</div>
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
        ${images.heatmap ? `<img src="${images.heatmap}" alt="Co-occurrence heatmap" style="max-height:80%; object-fit:contain;">` : '<p style="color:var(--muted);">[Image not found]</p>'}
        <p class="caption">Корелационна матрица на съвместно появяване на категории</p>
      </div>
      <div class="col">
        <div class="key-stats">
          <div class="key-stat"><span class="ks-val">282,371</span><span class="ks-label">сурови записа от 61 CSV файла</span></div>
          <div class="key-stat"><span class="ks-val">172,183</span><span class="ks-label">уникални доклада (след дедупликация)</span></div>
          <div class="key-stat"><span class="ks-val">39,894</span><span class="ks-label">стратифицирана извадка</span></div>
          <div class="key-stat"><span class="ks-val">31,850 / 8,044</span><span class="ks-label">train / test разделение</span></div>
          <div class="key-stat"><span class="ks-val">30.3\u00d7</span><span class="ks-label">дисбаланс между категории</span></div>
          <div class="key-stat"><span class="ks-val">13</span><span class="ks-label">категории аномалии (multi-label)</span></div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 4: FOUR APPROACHES ═══ -->
<section class="slide light" data-slide="3">
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
        <p>4-bit NF4 квантизация, LoRA r=16.<br>31,850 примера, 2 епохи, A100 GPU.<br>Тестван: Qwen3-8B</p>
      </div>
      <div class="card green">
        <h3>4. Classic ML (TF-IDF + XGBoost)</h3>
        <p>50K TF-IDF признака, 13 бинарни класификатора.<br>300 дървета, depth 6, scale_pos_weight.<br>Без GPU \u2014 бърз, евтин, стабилен.</p>
      </div>
    </div>
    <p class="footnote">Всички модели оценени на един и същ замразен тест сет (8,044 доклада)</p>
  </div>
</section>

<!-- ═══ SLIDE 5: MODELS USED ═══ -->
<section class="slide light" data-slide="4">
  <div class="header-bar"><h2>Използвани модели</h2></div>
  <div class="content" style="justify-content:center;">
    <table>
      <thead>
        <tr><th>Модел</th><th>Архитектура</th><th>Параметри</th><th>Подходи</th><th>Инфраструктура</th></tr>
      </thead>
      <tbody>
        <tr><td>Qwen3-8B</td><td>Dense Transformer</td><td>8B</td><td>ZS / FS / FT / Thinking</td><td>Modal L4 / A100 (vLLM)</td></tr>
        <tr><td>Mistral Large 3</td><td>MoE Transformer</td><td>675B (41B active)</td><td>ZS / FS</td><td>Mistral Batch API</td></tr>
        <tr><td>DeepSeek V3.2</td><td>MoE Transformer</td><td>671B</td><td>ZS / ZS+Thinking</td><td>DeepInfra API</td></tr>
        <tr><td>XGBoost</td><td>Gradient Boosted Trees</td><td>\u2014</td><td>TF-IDF baseline</td><td>Локално / Modal CPU</td></tr>
      </tbody>
    </table>
    <p style="text-align:center; font-size:12px; color:var(--muted); margin-top:4px;">ZS = Zero-Shot &nbsp;|&nbsp; FS = Few-Shot &nbsp;|&nbsp; FT = Fine-Tuned (QLoRA) &nbsp;|&nbsp; MoE = Mixture of Experts</p>
    <p style="text-align:center; font-size:13px; margin-top:12px;">Обща цена на експериментите: ~$53 (Modal GPU: ~$38 + DeepInfra API: ~$15 + Mistral: $0)</p>
  </div>
</section>

<!-- ═══ SLIDE 6: GRAND COMPARISON ═══ -->
<section class="slide light" data-slide="5">
  <div class="header-bar"><h2>Сравнение на всички модели (13 категории)</h2></div>
  <div class="content">
    <div class="fig-container">
      ${images.grand ? `<img src="${images.grand}" alt="Grand comparison">` : '<p style="color:var(--muted);">[Image not found]</p>'}
    </div>
    <div class="takeaway">
      Classic ML (Macro-F1 0.691) &gt; DeepSeek V3.2 + Thinking (0.681) &gt; Mistral Large 3 (0.658) &gt; Qwen3-8B Fine-Tuned (0.510)
    </div>
  </div>
</section>

<!-- ═══ SLIDE 7: BEST PER APPROACH ═══ -->
<section class="slide light" data-slide="6">
  <div class="header-bar"><h2>Най-добър модел по подход</h2></div>
  <div class="content">
    <div class="fig-container">
      ${images.approach ? `<img src="${images.approach}" alt="Approach summary">` : '<p style="color:var(--muted);">[Image not found]</p>'}
    </div>
    <div class="metrics-row">
      <div class="metric-card" style="border-top:3px solid var(--blue);"><div class="mc-approach" style="color:var(--blue);">Zero-Shot</div><div class="mc-detail">Mistral Large 3<br>Macro-F1: 0.658</div></div>
      <div class="metric-card" style="border-top:3px solid var(--orange);"><div class="mc-approach" style="color:var(--orange);">Few-Shot</div><div class="mc-detail">Mistral Large 3<br>Macro-F1: 0.640</div></div>
      <div class="metric-card" style="border-top:3px solid var(--red);"><div class="mc-approach" style="color:var(--red);">Fine-Tuned</div><div class="mc-detail">Qwen3-8B QLoRA<br>Macro-F1: 0.510</div></div>
      <div class="metric-card" style="border-top:3px solid var(--green);"><div class="mc-approach" style="color:var(--green);">Classic ML</div><div class="mc-detail">TF-IDF + XGBoost<br>Macro-F1: 0.691</div></div>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 8: CATEGORY HEATMAP ═══ -->
<section class="slide light" data-slide="7">
  <div class="header-bar"><h2>F1 по категория и модел (хийтмап)</h2></div>
  <div class="content">
    <div class="fig-container">
      ${images.categoryHeat ? `<img src="${images.categoryHeat}" alt="Category heatmap">` : '<p style="color:var(--muted);">[Image not found]</p>'}
    </div>
    <p class="caption">Стойности F1 за 13 категории \u00d7 15 модела. По-тъмно = по-висок F1. Classic ML доминира в повечето категории.</p>
  </div>
</section>

<!-- ═══ SLIDE 9: COST VS PERFORMANCE ═══ -->
<section class="slide light" data-slide="8">
  <div class="header-bar"><h2>Цена срещу представяне</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main">
        ${images.cost ? `<img src="${images.cost}" alt="Cost vs performance">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <div class="stat-card" style="border-top:4px solid var(--green);">
          <div class="value" style="color:var(--green); font-size:28px;">$0</div>
          <div class="label">Classic ML<br>(най-висок F1)</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--sky);">
          <div class="value" style="color:var(--sky); font-size:28px;">$6.73</div>
          <div class="label">DeepSeek V3.2<br>+ Thinking</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--red);">
          <div class="value" style="color:var(--red); font-size:28px;">~$19</div>
          <div class="label">Qwen3-8B<br>(всички опити)</div>
        </div>
      </div>
    </div>
    <p class="footnote">Classic ML е едновременно най-евтин и най-точен за 13-лейбъл задачата</p>
  </div>
</section>

<!-- ═══ SLIDE 10: SCALE VS TECHNIQUE ═══ -->
<section class="slide light" data-slide="9">
  <div class="header-bar"><h2>Мащаб срещу техника</h2></div>
  <div class="content">
    <div class="fig-with-side">
      <div class="fig-main">
        ${images.scale ? `<img src="${images.scale}" alt="Scale effect">` : '<p style="color:var(--muted);">[Image not found]</p>'}
      </div>
      <div class="fig-side">
        <div class="stat-card" style="border-top:4px solid var(--green);">
          <div class="value" style="color:var(--green); font-size:28px;">+0.058</div>
          <div class="label">Thinking ефект при 671B<br>(DeepSeek V3.2 parent)</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--orange);">
          <div class="value" style="color:var(--orange); font-size:28px;">+0.007</div>
          <div class="label">Thinking ефект при 8B<br>(Qwen3-8B FS taxonomy)</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--red);">
          <div class="value" style="color:var(--red); font-size:28px;">\u22120.003</div>
          <div class="label">Thinking при 48 лейбъла<br>(DeepSeek subcategory)</div>
        </div>
      </div>
    </div>
    <p class="footnote">Thinking mode работи при 671B за прости задачи, но вреди при сложни (48 лейбъла, 21.6% parse failures)</p>
  </div>
</section>

<!-- ═══ SLIDE 11: KEY FINDINGS ═══ -->
<section class="slide light" data-slide="10">
  <div class="header-bar"><h2>Ключови находки</h2></div>
  <div class="content" style="gap:4px;">
    <div class="finding" style="--c:var(--green);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--green);"></div>
      <h3 style="color:var(--green);">Classic ML доминира</h3>
      <p>TF-IDF + XGBoost (Macro-F1 0.691) превъзхожда всички LLM подходи за 13-лейбъл класификация. Без GPU, &lt;1 минута inference.</p>
    </div>
    <div class="finding" style="--c:var(--sky);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--sky);"></div>
      <h3 style="color:var(--sky);">Размерът на модела е решаващ</h3>
      <p>675B MoE (Mistral Large 3: 0.658) значително надминава 8B (Qwen3: 0.510). Fine-tuning на 8B не компенсира разликата.</p>
    </div>
    <div class="finding" style="--c:var(--orange);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--orange);"></div>
      <h3 style="color:var(--orange);">Taxonomy промптинг помага</h3>
      <p>Подкатегории + разграничителни подсказки подобряват Qwen3-8B с +0.073 Macro-F1. Особено полезно за малки модели.</p>
    </div>
    <div class="finding" style="--c:var(--red);"><div style="position:absolute;left:0;top:0;bottom:0;width:5px;border-radius:8px 0 0 8px;background:var(--red);"></div>
      <h3 style="color:var(--red);">Thinking mode \u2014 нюансиран ефект</h3>
      <p>При 671B: +0.058 F1 за 13 лейбъла, но \u22120.003 за 48 лейбъла (21.6% parse failures). При 8B: пренебрежим ефект (+0.007).</p>
    </div>
  </div>
</section>

<!-- ═══ SLIDE 12: CONCLUSION ═══ -->
<section class="slide dark" data-slide="11">
  <div class="content centered">
    <h1 style="font-size:32px;">Заключение</h1>
    <hr style="border:none; border-top:2px solid var(--sky); width:300px; margin:20px auto;">
    <p style="font-size:18px; line-height:1.5; max-width:800px; margin:0 auto;">
      За мултилейбълна класификация на авиационни доклади,<br>
      класическото ML (TF-IDF + XGBoost) остава най-ефективният подход \u2014<br>
      най-висок F1, нулева цена за inference, и най-бърз.
    </p>
    <h3 style="color:var(--sky); margin-top:28px; font-size:21px;">Бъдещи насоки</h3>
    <ul class="future-list">
      <li>Encoder-based модели (BERT, DeBERTa) за класификация</li>
      <li>RAG подход с вградени ASRS таксономии</li>
      <li>Мултимодално обучение (текст + структурирани полета)</li>
      <li>Поточна класификация за реално време</li>
    </ul>
    <hr style="border:none; border-top:2px solid var(--sky); width:300px; margin:28px auto 16px;">
    <p style="font-family:Georgia,serif; font-size:23px; color:var(--sky); font-weight:700;">Благодаря за вниманието!</p>
    <p style="color:var(--muted); font-size:13px; margin-top:8px;">Жарко Рашев &nbsp;|&nbsp; НБУ &nbsp;|&nbsp; 2026</p>
  </div>
</section>

<!-- ═══ CONTROLS ═══ -->
<div class="slide-counter" id="counter">1 / 12</div>
<div class="progress-bar" id="progress" style="width:8.33%;"></div>

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
    // Adapt counter color to slide background
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

  // Click navigation (left half = back, right half = forward)
  document.addEventListener('click', function(e) {
    if (overviewMode) {
      // Find which slide was clicked
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

  // Touch support
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
}

main();
