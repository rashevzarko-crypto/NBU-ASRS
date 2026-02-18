/**
 * NBU Thesis Defense Presentation Generator — PPTX
 * Generates defense_presentation.pptx with 12 slides for a 10-15 minute defense.
 *
 * Usage: node scripts/generate_defense_pptx.js
 * Output: defense_presentation.pptx in project root
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

// ─── CONSTANTS ───────────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, "..");
const RESULTS = path.join(ROOT, "results");

// Colors — NO # prefix (pptxgenjs requirement)
const C = {
  navy:   "0D1B2A",
  steel:  "1B4965",
  sky:    "5FA8D3",
  green:  "2CA02C",
  blue:   "1F77B4",
  orange: "FF7F0E",
  red:    "D62728",
  bg:     "F8F9FA",
  text:   "1E293B",
  muted:  "64748B",
  white:  "FFFFFF",
  black:  "000000",
  lightGray: "E2E8F0",
};

// Fonts
const F = {
  heading: "Georgia",
  body:    "Calibri",
  data:    "Consolas",
};

// Slide dimensions (inches, 16:9)
const W = 13.333;
const H = 7.5;

// ─── HELPERS ─────────────────────────────────────────────────────────────────

function makeShadow() {
  return { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.15 };
}

function loadImage(filename) {
  const p = path.join(RESULTS, filename);
  if (fs.existsSync(p)) {
    const buf = fs.readFileSync(p);
    return "image/png;base64," + buf.toString("base64");
  }
  console.warn(`WARNING: Missing image ${filename}`);
  return null;
}

/** Add a consistent slide header bar with title */
function addSlideHeader(slide, title) {
  // Header background bar
  slide.addShape("rect", {
    x: 0, y: 0, w: W, h: 0.9,
    fill: { color: C.steel },
  });
  // Title text
  slide.addText(title, {
    x: 0.6, y: 0.15, w: W - 1.2, h: 0.6,
    fontFace: F.heading, fontSize: 26, color: C.white,
    bold: true,
  });
}

/** Add a big stat callout box */
function addStatCallout(slide, x, y, w, h, value, label, accentColor) {
  // Background card
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C.white },
    rectRadius: 0.1,
    shadow: makeShadow(),
  });
  // Accent bar top
  slide.addShape("rect", {
    x: x, y: y, w: w, h: 0.06,
    fill: { color: accentColor },
  });
  // Value
  slide.addText(value, {
    x: x, y: y + 0.15, w: w, h: h * 0.5,
    fontFace: F.data, fontSize: 36, color: accentColor,
    bold: true, align: "center",
  });
  // Label
  slide.addText(label, {
    x: x + 0.1, y: y + h * 0.5, w: w - 0.2, h: h * 0.4,
    fontFace: F.body, fontSize: 13, color: C.muted,
    align: "center", valign: "top",
  });
}

/** Add a 2x2 card grid */
function addCard(slide, x, y, w, h, title, desc, accentColor) {
  // Card bg
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C.white },
    rectRadius: 0.1,
    shadow: makeShadow(),
  });
  // Left accent bar
  slide.addShape("rect", {
    x: x, y: y, w: 0.06, h: h,
    fill: { color: accentColor },
  });
  // Title
  slide.addText(title, {
    x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.4,
    fontFace: F.heading, fontSize: 16, color: C.text,
    bold: true,
  });
  // Description
  slide.addText(desc, {
    x: x + 0.2, y: y + 0.55, w: w - 0.4, h: h - 0.7,
    fontFace: F.body, fontSize: 12, color: C.muted,
    valign: "top",
  });
}

/** Add a full-width figure with caption below */
function addFullFigure(slide, imgData, caption, opts) {
  const defaults = { x: 0.6, y: 1.15, w: W - 1.2, h: 5.2 };
  const o = Object.assign({}, defaults, opts || {});
  if (imgData) {
    slide.addImage({
      data: imgData,
      x: o.x, y: o.y, w: o.w, h: o.h,
      sizing: { type: "contain", w: o.w, h: o.h },
    });
  } else {
    slide.addText("[Image not found]", {
      x: o.x, y: o.y, w: o.w, h: o.h,
      fontFace: F.body, fontSize: 14, color: C.muted,
      align: "center", valign: "middle",
    });
  }
  if (caption) {
    slide.addText(caption, {
      x: o.x, y: o.y + o.h + 0.1, w: o.w, h: 0.4,
      fontFace: F.body, fontSize: 11, color: C.muted,
      italic: true, align: "center",
    });
  }
}

// ─── SLIDE BUILDERS ──────────────────────────────────────────────────────────

function slideTitle(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.navy };

  // University name
  slide.addText("Нов български университет", {
    x: 0, y: 0.8, w: W, h: 0.5,
    fontFace: F.body, fontSize: 16, color: C.sky,
    align: "center", italic: true,
  });

  // Thesis title — Bulgarian
  slide.addText("Мултимодално дълбоко обучение за\nкласификация на инциденти по ВПП", {
    x: 1, y: 1.6, w: W - 2, h: 1.4,
    fontFace: F.heading, fontSize: 32, color: C.white,
    bold: true, align: "center", lineSpacingMultiple: 1.2,
  });

  // Subtitle — English
  slide.addText("Multi-Label Classification of Aviation Safety Reports\nUsing LLMs and Classic ML", {
    x: 1.5, y: 3.2, w: W - 3, h: 0.9,
    fontFace: F.body, fontSize: 18, color: C.sky,
    align: "center", lineSpacingMultiple: 1.2,
  });

  // Divider line
  slide.addShape("rect", {
    x: W / 2 - 2, y: 4.4, w: 4, h: 0.03,
    fill: { color: C.sky },
  });

  // Author info
  slide.addText("Жарко Рашев  |  Магистърска теза  |  2026", {
    x: 0, y: 4.8, w: W, h: 0.5,
    fontFace: F.body, fontSize: 16, color: C.muted,
    align: "center",
  });

  // Slide number footer (skip for title)
}

function slideResearchQuestion(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Изследователски въпрос");

  // Left column — research question
  slide.addText("Могат ли големите езикови модели (LLM) да конкурират класическото машинно обучение при мултилейбълна класификация на авиационни доклади за безопасност?", {
    x: 0.6, y: 1.3, w: 6.2, h: 1.8,
    fontFace: F.heading, fontSize: 20, color: C.text,
    italic: true, valign: "top", lineSpacingMultiple: 1.3,
  });

  // Sub-questions
  const subQs = [
    "Как се сравняват zero-shot, few-shot и fine-tuned LLM подходи?",
    "Каква е ролята на размера на модела (8B vs 675B)?",
    "Кога taxonomy-enriched промптинг помага?",
    "Какъв е компромисът между цена и качество?",
  ];
  subQs.forEach((q, i) => {
    slide.addText(`${i + 1}.  ${q}`, {
      x: 0.8, y: 3.3 + i * 0.55, w: 6, h: 0.5,
      fontFace: F.body, fontSize: 13, color: C.text,
    });
  });

  // Right column — stat callouts
  addStatCallout(slide, 7.8, 1.3, 2.2, 1.5, "172K", "ASRS доклада", C.sky);
  addStatCallout(slide, 10.3, 1.3, 2.2, 1.5, "13", "категории\nаномалии", C.green);
  addStatCallout(slide, 7.8, 3.2, 4.7, 1.5, "78%", "мулти-лейбъл доклади (2+ категории)", C.orange);

  // Bottom note
  slide.addText("NASA Aviation Safety Reporting System  |  1988\u20132024  |  Доброволни наративни доклади от пилоти и контрольори", {
    x: 0.6, y: 6.3, w: W - 1.2, h: 0.4,
    fontFace: F.body, fontSize: 11, color: C.muted,
    italic: true, align: "center",
  });
}

function slideDataset(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Данни \u2014 NASA ASRS");

  // Left — co-occurrence heatmap
  const img = loadImage("co_occurrence_heatmap.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.4, y: 1.2, w: 6.5, h: 5.4,
      sizing: { type: "contain", w: 6.5, h: 5.4 },
    });
  }

  // Right column — key stats
  const stats = [
    ["282,371", "сурови записа от 61 CSV файла"],
    ["172,183", "уникални доклада (след дедупликация по ACN)"],
    ["39,894", "стратифицирана извадка (MultilabelStratified)"],
    ["31,850 / 8,044", "train / test разделение"],
    ["30.3\u00d7", "дисбаланс (Deviation-Procedural vs Ground Excursion)"],
    ["13", "категории аномалии (multi-label)"],
  ];

  stats.forEach((s, i) => {
    slide.addText(s[0], {
      x: 7.3, y: 1.3 + i * 0.85, w: 2.2, h: 0.4,
      fontFace: F.data, fontSize: 18, color: C.sky,
      bold: true, align: "right",
    });
    slide.addText(s[1], {
      x: 9.6, y: 1.3 + i * 0.85, w: 3.2, h: 0.4,
      fontFace: F.body, fontSize: 12, color: C.text,
      valign: "middle",
    });
  });

  // Caption
  slide.addText("Фигура: Корелационна матрица на съвместно появяване на категории", {
    x: 0.4, y: 6.7, w: 6.5, h: 0.3,
    fontFace: F.body, fontSize: 10, color: C.muted,
    italic: true, align: "center",
  });
}

function slideFourApproaches(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Четири подхода за сравнение");

  const cardW = 5.8;
  const cardH = 2.5;
  const gap = 0.4;
  const startX = (W - 2 * cardW - gap) / 2;
  const startY = 1.3;

  addCard(slide, startX, startY, cardW, cardH,
    "1. Zero-Shot LLM",
    "Директна класификация без примери.\nМоделът разчита на предварително знание.\nТестван: Qwen3-8B, Mistral Large 3, DeepSeek V3.2",
    C.blue);

  addCard(slide, startX + cardW + gap, startY, cardW, cardH,
    "2. Few-Shot LLM",
    "3 примера на категория (39 общо) в промпта.\nПодбрани: кратки, еднолейбълни от train set.\nТестван: Qwen3-8B, Mistral Large 3",
    C.orange);

  addCard(slide, startX, startY + cardH + gap, cardW, cardH,
    "3. Fine-Tuned LLM (QLoRA)",
    "4-bit NF4 квантизация, LoRA r=16.\n31,850 примера, 2 епохи, A100 GPU.\nТестван: Qwen3-8B",
    C.red);

  addCard(slide, startX + cardW + gap, startY + cardH + gap, cardW, cardH,
    "4. Classic ML (TF-IDF + XGBoost)",
    "50K TF-IDF признака, 13 бинарни класификатора.\n300 дървета, depth 6, scale_pos_weight.\nБез GPU \u2014 бърз, евтин, стабилен.",
    C.green);

  // Footer note
  slide.addText("Всички модели оценени на един и същ замразен тест сет (8,044 доклада)", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.muted,
    italic: true, align: "center",
  });
}

function slideModelsUsed(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Използвани модели");

  const headerOpts = {
    fontFace: F.body, fontSize: 13, color: C.white, bold: true,
    align: "center", valign: "middle",
  };

  const rows = [
    ["Модел", "Архитектура", "Параметри", "Подходи", "Инфраструктура"],
    ["Qwen3-8B", "Dense Transformer", "8B", "ZS / FS / FT / Thinking", "Modal L4 / A100 (vLLM)"],
    ["Mistral Large 3", "MoE Transformer", "675B (41B active)", "ZS / FS", "Mistral Batch API"],
    ["DeepSeek V3.2", "MoE Transformer", "671B", "ZS / ZS+Thinking", "DeepInfra API"],
    ["XGBoost", "Gradient Boosted Trees", "\u2014", "TF-IDF baseline", "Локално / Modal CPU"],
  ];

  const colW = [2.8, 2.4, 2.0, 2.8, 2.6];
  const tableX = (W - colW.reduce((a, b) => a + b, 0)) / 2;
  const rowH = 0.65;
  const startY = 1.5;

  // Header row
  let cx = tableX;
  rows[0].forEach((cell, ci) => {
    slide.addShape("rect", {
      x: cx, y: startY, w: colW[ci], h: rowH,
      fill: { color: C.steel },
    });
    slide.addText(cell, {
      x: cx, y: startY, w: colW[ci], h: rowH,
      ...headerOpts,
    });
    cx += colW[ci];
  });

  // Data rows
  for (let ri = 1; ri < rows.length; ri++) {
    cx = tableX;
    const rowY = startY + ri * rowH;
    const bgColor = ri % 2 === 0 ? C.lightGray : C.white;
    rows[ri].forEach((cell, ci) => {
      slide.addShape("rect", {
        x: cx, y: rowY, w: colW[ci], h: rowH,
        fill: { color: bgColor },
        line: { color: C.lightGray, width: 0.5 },
      });
      slide.addText(cell, {
        x: cx + 0.1, y: rowY, w: colW[ci] - 0.2, h: rowH,
        fontFace: ci === 0 ? F.heading : F.body,
        fontSize: 12, color: C.text,
        bold: ci === 0,
        align: "center", valign: "middle",
      });
      cx += colW[ci];
    });
  }

  // Legend
  slide.addText("ZS = Zero-Shot  |  FS = Few-Shot  |  FT = Fine-Tuned (QLoRA)  |  MoE = Mixture of Experts", {
    x: 0, y: 5.5, w: W, h: 0.4,
    fontFace: F.body, fontSize: 11, color: C.muted,
    italic: true, align: "center",
  });

  // Cost summary
  slide.addText("Обща цена на експериментите: ~$53 (Modal GPU: ~$38 + DeepInfra API: ~$15 + Mistral: $0)", {
    x: 0, y: 6.0, w: W, h: 0.4,
    fontFace: F.body, fontSize: 12, color: C.text,
    align: "center",
  });
}

function slideGrandComparison(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Сравнение на всички модели (13 категории)");

  const img = loadImage("fig_grand_comparison.png");
  addFullFigure(slide, img, null, { y: 1.1, h: 5.0 });

  // Takeaway bar
  slide.addShape("rect", {
    x: 0.6, y: 6.2, w: W - 1.2, h: 0.7,
    fill: { color: C.white },
    rectRadius: 0.08,
    shadow: makeShadow(),
  });
  slide.addText("Classic ML (Macro-F1 0.691) > DeepSeek V3.2 + Thinking (0.681) > Mistral Large 3 (0.658) > Qwen3-8B Fine-Tuned (0.510)", {
    x: 0.8, y: 6.25, w: W - 1.6, h: 0.6,
    fontFace: F.body, fontSize: 13, color: C.text,
    align: "center", valign: "middle",
  });
}

function slideApproachSummary(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Най-добър модел по подход");

  const img = loadImage("fig_approach_summary.png");
  addFullFigure(slide, img, null, { y: 1.1, h: 4.6 });

  // Metrics row
  const metrics = [
    ["Zero-Shot", "Mistral Large 3", "0.658", C.blue],
    ["Few-Shot", "Mistral Large 3", "0.640", C.orange],
    ["Fine-Tuned", "Qwen3-8B QLoRA", "0.510", C.red],
    ["Classic ML", "TF-IDF + XGBoost", "0.691", C.green],
  ];

  const mw = 2.8;
  const mx = (W - metrics.length * mw - (metrics.length - 1) * 0.2) / 2;
  metrics.forEach((m, i) => {
    const bx = mx + i * (mw + 0.2);
    slide.addShape("rect", {
      x: bx, y: 5.9, w: mw, h: 1.0,
      fill: { color: C.white },
      rectRadius: 0.08,
      shadow: makeShadow(),
    });
    slide.addShape("rect", {
      x: bx, y: 5.9, w: mw, h: 0.05,
      fill: { color: m[3] },
    });
    slide.addText(m[0], {
      x: bx, y: 5.98, w: mw, h: 0.3,
      fontFace: F.heading, fontSize: 11, color: m[3],
      bold: true, align: "center",
    });
    slide.addText(`${m[1]}\nMacro-F1: ${m[2]}`, {
      x: bx + 0.1, y: 6.28, w: mw - 0.2, h: 0.55,
      fontFace: F.body, fontSize: 10, color: C.text,
      align: "center",
    });
  });
}

function slideCategoryHeatmap(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "F1 по категория и модел (хийтмап)");

  const img = loadImage("fig_category_heatmap.png");
  addFullFigure(slide, img,
    "Стойности F1 за 13 категории \u00d7 15 модела. По-тъмно = по-висок F1. Classic ML доминира в повечето категории.",
    { y: 1.1, h: 5.3 });
}

function slideCostPerformance(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Цена срещу представяне");

  const img = loadImage("fig_cost_vs_performance.png");
  addFullFigure(slide, img, null, { x: 0.4, y: 1.1, w: 8.5, h: 5.5 });

  // Callout cards on the right
  addStatCallout(slide, 9.3, 1.3, 3.4, 1.3, "$0", "Classic ML\n(най-висок F1)", C.green);
  addStatCallout(slide, 9.3, 2.9, 3.4, 1.3, "$6.73", "DeepSeek V3.2\n+ Thinking", C.sky);
  addStatCallout(slide, 9.3, 4.5, 3.4, 1.3, "~$19", "Qwen3-8B\n(всички опити)", C.red);

  slide.addText("Classic ML е едновременно най-евтин и най-точен за 13-лейбъл задачата", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.text,
    italic: true, align: "center",
  });
}

function slideScaleEffect(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Мащаб срещу техника");

  // Left — figure
  const img = loadImage("fig_scale_effect.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.4, y: 1.2, w: 7.5, h: 5.2,
      sizing: { type: "contain", w: 7.5, h: 5.2 },
    });
  }

  // Right — callout stats
  addStatCallout(slide, 8.3, 1.4, 4.3, 1.3, "+0.058", "Thinking ефект при 671B\n(DeepSeek V3.2 parent)", C.green);
  addStatCallout(slide, 8.3, 3.0, 4.3, 1.3, "+0.007", "Thinking ефект при 8B\n(Qwen3-8B FS taxonomy)", C.orange);
  addStatCallout(slide, 8.3, 4.6, 4.3, 1.3, "\u22120.003", "Thinking при 48 лейбъла\n(DeepSeek subcategory)", C.red);

  slide.addText("Thinking mode работи при 671B за прости задачи, но вреди при сложни (48 лейбъла, 21.6% parse failures)", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.text,
    italic: true, align: "center",
  });
}

function slideKeyFindings(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Ключови находки");

  const findings = [
    {
      color: C.green,
      title: "Classic ML доминира",
      text: "TF-IDF + XGBoost (Macro-F1 0.691) превъзхожда всички LLM подходи за 13-лейбъл класификация. Без GPU, <1 минута inference.",
    },
    {
      color: C.sky,
      title: "Размерът на модела е решаващ",
      text: "675B MoE (Mistral Large 3: 0.658) значително надминава 8B (Qwen3: 0.510). Fine-tuning на 8B не компенсира разликата.",
    },
    {
      color: C.orange,
      title: "Taxonomy промптинг помага",
      text: "Подкатегории + разграничителни подсказки подобряват Qwen3-8B с +0.073 Macro-F1. Особено полезно за малки модели.",
    },
    {
      color: C.red,
      title: "Thinking mode \u2014 нюансиран ефект",
      text: "При 671B: +0.058 F1 за 13 лейбъла, но \u22120.003 за 48 лейбъла (21.6% parse failures). При 8B: пренебрежим ефект (+0.007).",
    },
  ];

  const cardW = W - 1.2;
  const cardH = 0.95;
  const startY = 1.2;
  const gap = 0.15;

  findings.forEach((f, i) => {
    const cy = startY + i * (cardH + gap);
    // Card bg
    slide.addShape("rect", {
      x: 0.6, y: cy, w: cardW, h: cardH,
      fill: { color: C.white },
      rectRadius: 0.08,
      shadow: makeShadow(),
    });
    // Left accent
    slide.addShape("rect", {
      x: 0.6, y: cy, w: 0.06, h: cardH,
      fill: { color: f.color },
    });
    // Title
    slide.addText(f.title, {
      x: 0.85, y: cy + 0.05, w: 3.5, h: 0.35,
      fontFace: F.heading, fontSize: 14, color: f.color,
      bold: true,
    });
    // Text
    slide.addText(f.text, {
      x: 0.85, y: cy + 0.35, w: cardW - 0.5, h: cardH - 0.4,
      fontFace: F.body, fontSize: 11, color: C.text,
      valign: "top",
    });
  });
}

function slideConclusion(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.navy };

  // Main conclusion
  slide.addText("Заключение", {
    x: 0, y: 0.8, w: W, h: 0.6,
    fontFace: F.heading, fontSize: 30, color: C.white,
    bold: true, align: "center",
  });

  // Divider
  slide.addShape("rect", {
    x: W / 2 - 2, y: 1.6, w: 4, h: 0.03,
    fill: { color: C.sky },
  });

  // Main statement
  slide.addText("За мултилейбълна класификация на авиационни доклади,\nкласическото ML (TF-IDF + XGBoost) остава най-ефективният подход \u2014\nнай-висок F1, нулева цена за inference, и най-бърз.", {
    x: 1, y: 2.0, w: W - 2, h: 1.4,
    fontFace: F.body, fontSize: 18, color: C.white,
    align: "center", lineSpacingMultiple: 1.3,
  });

  // Future work
  slide.addText("Бъдещи насоки", {
    x: 0, y: 3.8, w: W, h: 0.5,
    fontFace: F.heading, fontSize: 20, color: C.sky,
    bold: true, align: "center",
  });

  const futureWork = [
    "Encoder-based модели (BERT, DeBERTa) за класификация",
    "RAG подход с вградени ASRS таксономии",
    "Мултимодално обучение (текст + структурирани полета)",
    "Поточна класификация за реално време",
  ];

  futureWork.forEach((item, i) => {
    slide.addText(`\u2022  ${item}`, {
      x: 2.5, y: 4.4 + i * 0.45, w: W - 5, h: 0.4,
      fontFace: F.body, fontSize: 14, color: C.white,
      lineSpacingMultiple: 1.2,
    });
  });

  // Thank you
  slide.addShape("rect", {
    x: W / 2 - 2, y: 6.2, w: 4, h: 0.03,
    fill: { color: C.sky },
  });
  slide.addText("Благодаря за вниманието!", {
    x: 0, y: 6.4, w: W, h: 0.6,
    fontFace: F.heading, fontSize: 22, color: C.sky,
    bold: true, align: "center",
  });

  slide.addText("Жарко Рашев  |  НБУ  |  2026", {
    x: 0, y: 7.0, w: W, h: 0.3,
    fontFace: F.body, fontSize: 12, color: C.muted,
    align: "center",
  });
}

// ─── SLIDE NUMBERS ───────────────────────────────────────────────────────────

function addSlideNumbers(pptx) {
  // pptxgenjs doesn't have per-slide footers easily, so we add text to each slide
  const slides = pptx.slides;
  slides.forEach((slide, i) => {
    if (i === 0 || i === slides.length - 1) return; // skip title and conclusion
    slide.addText(`${i + 1} / ${slides.length}`, {
      x: W - 1.5, y: H - 0.45, w: 1.2, h: 0.3,
      fontFace: F.body, fontSize: 10, color: C.muted,
      align: "right",
    });
  });
}

// ─── MAIN ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("Generating defense_presentation.pptx...");

  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_16x9";
  pptx.author = "Zharko Rashev";
  pptx.title = "Multi-Label Classification of Aviation Safety Reports";
  pptx.subject = "NBU Master's Thesis Defense";

  // Build all 12 slides
  slideTitle(pptx);            // 1
  slideResearchQuestion(pptx); // 2
  slideDataset(pptx);          // 3
  slideFourApproaches(pptx);   // 4
  slideModelsUsed(pptx);       // 5
  slideGrandComparison(pptx);  // 6
  slideApproachSummary(pptx);  // 7
  slideCategoryHeatmap(pptx);  // 8
  slideCostPerformance(pptx);  // 9
  slideScaleEffect(pptx);      // 10
  slideKeyFindings(pptx);      // 11
  slideConclusion(pptx);       // 12

  addSlideNumbers(pptx);

  const outPath = path.join(ROOT, "defense_presentation.pptx");
  await pptx.writeFile({ fileName: outPath });
  console.log(`Written: ${outPath}`);
  console.log(`Total slides: ${pptx.slides.length}`);
}

main().catch(err => {
  console.error("Error generating PPTX:", err);
  process.exit(1);
});
