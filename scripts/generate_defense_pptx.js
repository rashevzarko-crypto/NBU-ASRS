/**
 * NBU Thesis Defense Presentation Generator — PPTX
 * Generates defense_presentation.pptx with 20 slides for a 15-20 minute defense.
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
  codeGray:  "F1F5F9",
  greenBg:   "DCFCE7",
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

// ─── HELPERS (existing) ─────────────────────────────────────────────────────

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
  slide.addShape("rect", {
    x: 0, y: 0, w: W, h: 0.9,
    fill: { color: C.steel },
  });
  slide.addText(title, {
    x: 0.6, y: 0.15, w: W - 1.2, h: 0.6,
    fontFace: F.heading, fontSize: 26, color: C.white,
    bold: true,
  });
}

/** Add a big stat callout box */
function addStatCallout(slide, x, y, w, h, value, label, accentColor) {
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C.white },
    rectRadius: 0.1,
    shadow: makeShadow(),
  });
  slide.addShape("rect", {
    x: x, y: y, w: w, h: 0.06,
    fill: { color: accentColor },
  });
  slide.addText(value, {
    x: x, y: y + 0.15, w: w, h: h * 0.5,
    fontFace: F.data, fontSize: 36, color: accentColor,
    bold: true, align: "center",
  });
  slide.addText(label, {
    x: x + 0.1, y: y + h * 0.5, w: w - 0.2, h: h * 0.4,
    fontFace: F.body, fontSize: 13, color: C.muted,
    align: "center", valign: "top",
  });
}

/** Add a card with left accent bar */
function addCard(slide, x, y, w, h, title, desc, accentColor) {
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C.white },
    rectRadius: 0.1,
    shadow: makeShadow(),
  });
  slide.addShape("rect", {
    x: x, y: y, w: 0.06, h: h,
    fill: { color: accentColor },
  });
  slide.addText(title, {
    x: x + 0.2, y: y + 0.15, w: w - 0.4, h: 0.4,
    fontFace: F.heading, fontSize: 16, color: C.text,
    bold: true,
  });
  slide.addText(desc, {
    x: x + 0.2, y: y + 0.55, w: w - 0.4, h: h - 0.7,
    fontFace: F.body, fontSize: 12, color: C.muted,
    valign: "top",
  });
}

/** Add a full-width figure with optional caption */
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

// ─── HELPERS (new) ──────────────────────────────────────────────────────────

/** Add a pipeline step box */
function addPipelineBox(slide, x, y, w, h, title, subtitle, bgColor) {
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: bgColor },
    rectRadius: 0.1,
    shadow: makeShadow(),
  });
  slide.addText(title, {
    x: x + 0.1, y: y + 0.1, w: w - 0.2, h: h * 0.5,
    fontFace: F.heading, fontSize: 14, color: C.white,
    bold: true, align: "center", valign: "middle",
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: x + 0.1, y: y + h * 0.5, w: w - 0.2, h: h * 0.45,
      fontFace: F.body, fontSize: 10, color: C.white,
      align: "center", valign: "top",
    });
  }
}

/** Add a horizontal arrow between pipeline boxes */
function addArrow(slide, x, y, w) {
  slide.addShape("line", {
    x: x, y: y, w: w, h: 0,
    line: { color: C.muted, width: 2, endArrowType: "triangle" },
  });
}

/** Add a code/prompt block with label */
function addCodeBox(slide, x, y, w, h, text, label) {
  if (label) {
    slide.addText(label, {
      x: x, y: y - 0.3, w: w, h: 0.28,
      fontFace: F.body, fontSize: 11, color: C.text,
      bold: true, valign: "bottom",
    });
  }
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C.codeGray },
    rectRadius: 0.08,
    line: { color: C.lightGray, width: 1 },
  });
  slide.addText(text, {
    x: x + 0.15, y: y + 0.08, w: w - 0.3, h: h - 0.16,
    fontFace: F.data, fontSize: 9, color: C.text,
    valign: "top", lineSpacingMultiple: 1.15,
  });
}

/** Generic table renderer */
function addTable(slide, headers, rows, opts) {
  const o = Object.assign({
    y: 1.5, rowHeight: 0.5, headerColor: C.steel,
    fontSize: 11, headerFontSize: 12, centerX: true,
  }, opts || {});

  const colWidths = o.colWidths || headers.map(() => (W - 1.2) / headers.length);
  const tableW = colWidths.reduce((a, b) => a + b, 0);
  const startX = o.x !== undefined ? o.x : (o.centerX ? (W - tableW) / 2 : 0.6);

  // Header row
  let cx = startX;
  headers.forEach((h, ci) => {
    slide.addShape("rect", {
      x: cx, y: o.y, w: colWidths[ci], h: o.rowHeight,
      fill: { color: o.headerColor },
    });
    slide.addText(h, {
      x: cx, y: o.y, w: colWidths[ci], h: o.rowHeight,
      fontFace: F.body, fontSize: o.headerFontSize, color: C.white,
      bold: true, align: "center", valign: "middle",
    });
    cx += colWidths[ci];
  });

  // Data rows
  for (let ri = 0; ri < rows.length; ri++) {
    cx = startX;
    const rowY = o.y + (ri + 1) * o.rowHeight;
    const bgColor = ri % 2 === 0 ? C.white : C.lightGray;
    rows[ri].forEach((cell, ci) => {
      slide.addShape("rect", {
        x: cx, y: rowY, w: colWidths[ci], h: o.rowHeight,
        fill: { color: bgColor },
        line: { color: C.lightGray, width: 0.5 },
      });
      slide.addText(cell, {
        x: cx + 0.08, y: rowY, w: colWidths[ci] - 0.16, h: o.rowHeight,
        fontFace: ci === 0 ? F.heading : F.body,
        fontSize: o.fontSize, color: C.text,
        bold: ci === 0,
        align: ci === 0 ? "left" : "center",
        valign: "middle",
      });
      cx += colWidths[ci];
    });
  }
}

/** Labeled box for conceptual diagrams */
function addDiagramBox(slide, x, y, w, h, title, lines, opts) {
  const o = Object.assign({ accentColor: C.steel, titleSize: 13, lineSize: 11 }, opts || {});
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C.white },
    rectRadius: 0.08,
    line: { color: o.accentColor, width: 1.5 },
  });
  slide.addShape("rect", {
    x: x, y: y, w: w, h: 0.05,
    fill: { color: o.accentColor },
  });
  slide.addText(title, {
    x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.35,
    fontFace: F.heading, fontSize: o.titleSize, color: o.accentColor,
    bold: true,
  });
  if (lines) {
    slide.addText(lines, {
      x: x + 0.12, y: y + 0.42, w: w - 0.24, h: h - 0.52,
      fontFace: F.body, fontSize: o.lineSize, color: C.text,
      valign: "top", lineSpacingMultiple: 1.2,
    });
  }
}

// ─── SLIDE 1: TITLE ─────────────────────────────────────────────────────────

function slideTitle(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.navy };

  // University name
  slide.addText("Нов български университет", {
    x: 0, y: 0.6, w: W, h: 0.4,
    fontFace: F.body, fontSize: 16, color: C.sky,
    align: "center", italic: true,
  });

  // Department
  slide.addText("Департамент по информатика", {
    x: 0, y: 1.0, w: W, h: 0.35,
    fontFace: F.body, fontSize: 13, color: C.muted,
    align: "center",
  });

  // Thesis title — Bulgarian
  slide.addText("Мултимодално дълбоко обучение за\nкласификация на инциденти по ВПП", {
    x: 1, y: 1.7, w: W - 2, h: 1.3,
    fontFace: F.heading, fontSize: 32, color: C.white,
    bold: true, align: "center", lineSpacingMultiple: 1.2,
  });

  // English subtitle
  slide.addText("Multi-Label Classification of Aviation Safety Reports\nUsing LLMs and Classic ML", {
    x: 1.5, y: 3.2, w: W - 3, h: 0.85,
    fontFace: F.body, fontSize: 18, color: C.sky,
    align: "center", lineSpacingMultiple: 1.2,
  });

  // Divider
  slide.addShape("rect", {
    x: W / 2 - 2, y: 4.3, w: 4, h: 0.03,
    fill: { color: C.sky },
  });

  // Author
  slide.addText("Зарко Рашев  Ф№ F98363", {
    x: 0, y: 4.7, w: W, h: 0.4,
    fontFace: F.body, fontSize: 16, color: C.white,
    align: "center",
  });

  // Advisor
  slide.addText("Научен ръководител: доц. д-р Стоян Мишев", {
    x: 0, y: 5.15, w: W, h: 0.35,
    fontFace: F.body, fontSize: 14, color: C.muted,
    align: "center",
  });

  // Year
  slide.addText("Магистърска теза  |  2026", {
    x: 0, y: 5.6, w: W, h: 0.35,
    fontFace: F.body, fontSize: 13, color: C.muted,
    align: "center",
  });
}

// ─── SLIDE 2: RESEARCH QUESTION ─────────────────────────────────────────────

function slideResearchQuestion(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Изследователски въпрос");

  // Left — main RQ
  slide.addText("Могат ли големите езикови модели (LLM) да конкурират класическото машинно обучение при мултилейбълна класификация на авиационни доклади за безопасност?", {
    x: 0.6, y: 1.3, w: 6.2, h: 1.6,
    fontFace: F.heading, fontSize: 19, color: C.text,
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
      x: 0.8, y: 3.1 + i * 0.52, w: 6, h: 0.48,
      fontFace: F.body, fontSize: 13, color: C.text,
    });
  });

  // Right — stat callouts (3 across top + 1 wide)
  addStatCallout(slide, 7.4, 1.3, 1.8, 1.3, "172K", "ASRS доклада", C.sky);
  addStatCallout(slide, 9.4, 1.3, 1.5, 1.3, "13", "категории", C.green);
  addStatCallout(slide, 11.1, 1.3, 1.6, 1.3, "22", "експеримента", C.orange);
  addStatCallout(slide, 7.4, 3.0, 5.3, 1.3, "78%", "мулти-лейбъл доклади (2+ категории)", C.red);

  // Bottom note
  slide.addText("NASA Aviation Safety Reporting System  |  1988\u20132024  |  Доброволни наративни доклади от пилоти и контрольори", {
    x: 0.6, y: 6.3, w: W - 1.2, h: 0.4,
    fontFace: F.body, fontSize: 11, color: C.muted,
    italic: true, align: "center",
  });
}

// ─── SLIDE 3: DATASET ───────────────────────────────────────────────────────

function slideDataset(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Данни \u2014 NASA ASRS");

  // Left — heatmap
  const img = loadImage("co_occurrence_heatmap.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.4, y: 1.1, w: 5.8, h: 4.8,
      sizing: { type: "contain", w: 5.8, h: 4.8 },
    });
  }
  slide.addText("Корелационна матрица на съвместно появяване", {
    x: 0.4, y: 5.95, w: 5.8, h: 0.3,
    fontFace: F.body, fontSize: 10, color: C.muted,
    italic: true, align: "center",
  });

  // Right — mini category table (top 5 + bottom 3)
  const catHeaders = ["Категория", "%"];
  const catRows = [
    ["Deviation-Procedural", "65.4%"],
    ["Aircraft Equipment Problem", "28.6%"],
    ["Conflict", "26.9%"],
    ["Inflight Event/Encounter", "22.5%"],
    ["ATC Issue", "17.1%"],
    ["\u2026", ""],
    ["Airspace Violation", "4.0%"],
    ["Deviation-Speed", "2.9%"],
    ["Ground Excursion", "2.2%"],
  ];
  addTable(slide, catHeaders, catRows, {
    x: 6.6, y: 1.1, colWidths: [3.8, 1.0],
    rowHeight: 0.42, fontSize: 10, headerFontSize: 11,
  });

  // Key stats below the table
  const stats = [
    ["31,850 / 8,044", "train / test разделение"],
    ["30.3\u00d7", "дисбаланс между категории"],
    ["78%", "доклади с 2+ категории"],
  ];
  stats.forEach((s, i) => {
    slide.addText(s[0], {
      x: 6.6, y: 5.3 + i * 0.48, w: 1.8, h: 0.4,
      fontFace: F.data, fontSize: 14, color: C.sky,
      bold: true, align: "right",
    });
    slide.addText(s[1], {
      x: 8.5, y: 5.3 + i * 0.48, w: 3.5, h: 0.4,
      fontFace: F.body, fontSize: 11, color: C.text,
      valign: "middle",
    });
  });
}

// ─── SLIDE 4: MULTI-LABEL EXPLAINED (NEW) ───────────────────────────────────

function slideMultiLabelExplained(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Какво е мулти-лейбъл класификация?");

  // Left: multi-class vs multi-label comparison
  addDiagramBox(slide, 0.6, 1.2, 3.5, 1.6,
    "Multi-Class (стандартна)",
    "1 доклад \u2192 1 категория\nВзаимно изключващи се класове\nПример: спам / не-спам",
    { accentColor: C.muted });

  addDiagramBox(slide, 0.6, 3.1, 3.5, 1.6,
    "Multi-Label (нашата задача)",
    "1 доклад \u2192 1\u201313 категории\n78% от докладите имат 2+ лейбъла\nМедиана: 2 категории на доклад",
    { accentColor: C.sky });

  // Right: example ASRS report
  addDiagramBox(slide, 4.5, 1.2, 5.2, 3.5,
    "Примерен ASRS доклад",
    "\"While descending through FL240, we received a TCAS RA\nfor traffic at our 12 o'clock. We followed the RA guidance\nand deviated from our assigned altitude. ATC was notified\nimmediately but there was a delay in acknowledgment...\"\n\n\u2192 Conflict\n\u2192 Deviation - Altitude\n\u2192 ATC Issue",
    { accentColor: C.blue, lineSize: 10 });

  // Bottom: metrics explanation
  const metricsY = 5.1;
  const mw = 3.7;
  const mx = (W - 3 * mw - 0.4) / 2;

  addDiagramBox(slide, mx, metricsY, mw, 1.5,
    "Macro-F1",
    "Средно F1 на всички 13 категории.\nТретира всяка категория еднакво.\nЧувствителна към редки класове.",
    { accentColor: C.green, titleSize: 12, lineSize: 10 });

  addDiagramBox(slide, mx + mw + 0.2, metricsY, mw, 1.5,
    "Micro-F1",
    "Глобално F1 от всички предикции.\nПретегля по брой примери.\nПо-високо при чести категории.",
    { accentColor: C.orange, titleSize: 12, lineSize: 10 });

  addDiagramBox(slide, mx + 2 * (mw + 0.2), metricsY, mw, 1.5,
    "Macro-AUC (ROC)",
    "Качество на ранжирането.\nНезависимо от прага на решение.\n1.0 = перфектно, 0.5 = случайно.",
    { accentColor: C.sky, titleSize: 12, lineSize: 10 });
}

// ─── SLIDE 5: FOUR APPROACHES ───────────────────────────────────────────────

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
    "4-bit NF4 квантизация, LoRA r=16.\n31,850 примера, 2 епохи, A100 GPU.\nТестван: Qwen3-8B (Ministral 8B \u2014 FP8 проблем)",
    C.red);

  addCard(slide, startX + cardW + gap, startY + cardH + gap, cardW, cardH,
    "4. Classic ML (TF-IDF + XGBoost)",
    "50K TF-IDF признака, 13 бинарни класификатора.\n300 дървета, depth 6, scale_pos_weight.\nБез GPU \u2014 бърз, евтин, стабилен.",
    C.green);

  slide.addText("Всички модели оценени на един и същ замразен тест сет (8,044 доклада)", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.muted,
    italic: true, align: "center",
  });
}

// ─── SLIDE 6: CLASSIC ML PIPELINE (NEW) ─────────────────────────────────────

function slideClassicMLPipeline(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Classic ML \u2014 TF-IDF + XGBoost Pipeline");

  // Pipeline flow: Narrative → TF-IDF → Sparse Matrix → 13× XGBoost → Predictions
  const pipeY = 1.5;
  const boxH = 1.2;
  const boxW = 2.0;
  const arrowW = 0.5;
  const startX = 0.5;

  addPipelineBox(slide, startX, pipeY, boxW, boxH,
    "Наратив", "Текст от пилот\n(~200 думи)", C.steel);
  addArrow(slide, startX + boxW + 0.05, pipeY + boxH / 2, arrowW);

  addPipelineBox(slide, startX + boxW + arrowW + 0.15, pipeY, boxW, boxH,
    "TF-IDF", "50K признака\nngram(1,2)", C.blue);
  addArrow(slide, startX + 2 * boxW + arrowW + 0.2, pipeY + boxH / 2, arrowW);

  addPipelineBox(slide, startX + 2 * (boxW + arrowW) + 0.3, pipeY, 2.3, boxH,
    "Разредена матрица", "31,850 \u00d7 50,000\nsublinear TF", C.sky);
  addArrow(slide, startX + 2 * (boxW + arrowW) + 2.6 + 0.05, pipeY + boxH / 2, arrowW);

  addPipelineBox(slide, startX + 2 * (boxW + arrowW) + 2.6 + arrowW + 0.15, pipeY, boxW, boxH,
    "13\u00d7 XGBoost", "Бинарен\nкласификатор", C.orange);
  addArrow(slide, startX + 3 * (boxW + arrowW) + 2.6 + 0.2, pipeY + boxH / 2, arrowW);

  addPipelineBox(slide, startX + 3 * (boxW + arrowW) + 2.6 + arrowW + 0.3, pipeY, boxW, boxH,
    "Предикции", "13 вероятности\n(multi-label)", C.green);

  // Explanation boxes below
  const expY = 3.2;
  const expH = 1.8;
  const expW = 3.8;
  const expGap = 0.3;
  const expStartX = (W - 3 * expW - 2 * expGap) / 2;

  addDiagramBox(slide, expStartX, expY, expW, expH,
    "TF-IDF Vectorizer",
    "Term Frequency \u2013 Inverse Document Frequency\nПретегля думи по важност в документа\nспрямо целия корпус. Улавя domain-\nспецифични n-грами: \"runway incursion\",\n\"altitude deviation\", \"TCAS RA\".",
    { accentColor: C.blue, titleSize: 12, lineSize: 10 });

  addDiagramBox(slide, expStartX + expW + expGap, expY, expW, expH,
    "XGBoost (One-vs-Rest)",
    "13 независими бинарни класификатора.\nВсеки има собствен scale_pos_weight\nза справяне с дисбаланса.\n300 дървета, depth 6, learning rate 0.1.\ntree_method=hist за бързина.",
    { accentColor: C.orange, titleSize: 12, lineSize: 10 });

  addDiagramBox(slide, expStartX + 2 * (expW + expGap), expY, expW, expH,
    "Защо работи?",
    "Per-label оптимизация: всяка категория\nима собствен модел и праг.\nГрадирани вероятности (0\u20131) вместо\nда/не. Бърз: <1 минута inference\nна 8K доклада без GPU.",
    { accentColor: C.green, titleSize: 12, lineSize: 10 });

  slide.addText("Хиперпараметрично търсене (3-fold CV, 8 TF-IDF + 3 модела) потвърждава: baseline е оптимален (\u0394 Macro-F1 < 0.005)", {
    x: 0.6, y: 6.8, w: W - 1.2, h: 0.35,
    fontFace: F.body, fontSize: 11, color: C.muted,
    italic: true, align: "center",
  });
}

// ─── SLIDE 7: LLM ARCHITECTURE (NEW) ────────────────────────────────────────

function slideLLMArchitecture(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "LLM архитектура \u2014 Dense vs MoE");

  // Left: Dense vs MoE diagram
  addDiagramBox(slide, 0.6, 1.2, 4.0, 2.2,
    "Dense Transformer",
    "Всички параметри активни при inference.\nQwen3-8B: 8B параметра, всички активни.\nПо-малък, по-бърз, по-евтин.\nЛесен за fine-tuning (QLoRA).",
    { accentColor: C.blue });

  addDiagramBox(slide, 0.6, 3.7, 4.0, 2.2,
    "Mixture of Experts (MoE)",
    "Маршрутизатор избира подмножество експерти.\nMistral Large 3: 675B (41B активни).\nDeepSeek V3.2: 671B MoE.\nПо-мощен, но изисква API достъп.",
    { accentColor: C.orange });

  // Right: Models table
  const tHeaders = ["Модел", "Архитектура", "Параметри", "Лиценз"];
  const tRows = [
    ["Qwen3-8B", "Dense", "8B", "Apache 2.0"],
    ["Mistral Large 3", "MoE", "675B (41B act.)", "Apache 2.0"],
    ["DeepSeek V3.2", "MoE", "671B", "MIT"],
    ["Ministral 8B", "Dense (FP8)", "8B", "Apache 2.0"],
  ];
  addTable(slide, tHeaders, tRows, {
    x: 5.2, y: 1.2, colWidths: [2.4, 1.5, 2.0, 1.7],
    rowHeight: 0.55, fontSize: 11, headerFontSize: 12,
    centerX: false,
  });

  // Infrastructure note
  addDiagramBox(slide, 5.2, 3.7, 7.5, 2.2,
    "Инфраструктура",
    "Qwen3-8B: Modal GPU (L4 24GB inference, A100 80GB training)\n \u2192 vLLM за бързо batch inference, enable_thinking=False\n\nMistral Large 3: Mistral Batch API (безплатен план)\n \u2192 8K заявки за ~5 мин, без rate limit проблеми\n\nDeepSeek V3.2: DeepInfra API (prefix caching 62\u201382%)\n \u2192 50 конкурентни заявки, aiohttp",
    { accentColor: C.steel, titleSize: 13, lineSize: 10 });

  slide.addText("Общо: 22 експеримента на 4 модела (3 LLM + 1 Classic ML)", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.muted,
    italic: true, align: "center",
  });
}

// ─── SLIDE 8: QLORA FINE-TUNING (NEW) ───────────────────────────────────────

function slideQLoRA(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "QLoRA Fine-Tuning");

  // Left: visual — frozen model + LoRA adapters
  // Big frozen model box
  slide.addShape("rect", {
    x: 0.6, y: 1.3, w: 5.0, h: 3.5,
    fill: { color: C.lightGray },
    rectRadius: 0.1,
    line: { color: C.muted, width: 1, dashType: "dash" },
  });
  slide.addText("Qwen3-8B\n(замразен, 4-bit NF4)", {
    x: 0.6, y: 1.5, w: 5.0, h: 0.8,
    fontFace: F.heading, fontSize: 16, color: C.muted,
    align: "center",
  });

  // LoRA adapter boxes inside
  const loraBoxes = [
    { x: 1.2, y: 2.6, label: "LoRA A\n(q_proj)" },
    { x: 3.2, y: 2.6, label: "LoRA B\n(v_proj)" },
  ];
  loraBoxes.forEach(b => {
    slide.addShape("rect", {
      x: b.x, y: b.y, w: 1.8, h: 1.0,
      fill: { color: C.white },
      rectRadius: 0.08,
      line: { color: C.red, width: 2 },
    });
    slide.addText(b.label, {
      x: b.x + 0.1, y: b.y + 0.1, w: 1.6, h: 0.8,
      fontFace: F.data, fontSize: 12, color: C.red,
      bold: true, align: "center", valign: "middle",
    });
  });

  slide.addText("r=16, \u03b1=16, dropout=0.05", {
    x: 0.6, y: 3.85, w: 5.0, h: 0.35,
    fontFace: F.data, fontSize: 11, color: C.text,
    align: "center",
  });

  slide.addText("\u2764 Trainable: 0.2% от параметрите", {
    x: 0.6, y: 4.25, w: 5.0, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.red,
    bold: true, align: "center",
  });

  // Right: config table
  const cfgHeaders = ["Параметър", "Стойност"];
  const cfgRows = [
    ["Квантизация", "4-bit NF4 (double quant)"],
    ["LoRA rank", "r=16, \u03b1=16"],
    ["Target modules", "q_proj, v_proj"],
    ["Training data", "31,850 примера"],
    ["Epochs", "2 (3,982 стъпки)"],
    ["Batch size", "4 (grad accum \u00d74)"],
    ["Learning rate", "2e-5 (cosine)"],
    ["Optimizer", "paged_adamw_8bit"],
    ["GPU", "A100 80GB (Modal)"],
    ["Времетраене", "3h 47min"],
    ["Финална загуба", "1.691 (66.8% acc)"],
  ];
  addTable(slide, cfgHeaders, cfgRows, {
    x: 6.2, y: 1.2, colWidths: [2.4, 3.4],
    rowHeight: 0.42, fontSize: 10, headerFontSize: 11,
    centerX: false,
  });

  // Ministral FP8 note
  slide.addShape("rect", {
    x: 0.6, y: 5.3, w: W - 1.2, h: 0.9,
    fill: { color: "FFF3CD" },
    rectRadius: 0.08,
  });
  slide.addText("\u26a0 Ministral 8B: Mistral3ForConditionalGeneration (мултимодален) с FP8 квантизация \u2014 не позволява QLoRA (4-bit NF4). Fine-tuning не подобри резултатите (Macro-F1: 0.489 vs 0.491 zero-shot). Заменен с Qwen3-8B.", {
    x: 0.8, y: 5.4, w: W - 1.6, h: 0.7,
    fontFace: F.body, fontSize: 11, color: C.text,
    valign: "middle",
  });
}

// ─── SLIDE 9: PROMPT ENGINEERING (NEW) ──────────────────────────────────────

function slidePromptEngineering(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Prompt Engineering \u2014 Taxonomy-Enriched");

  // Left: Basic prompt
  addCodeBox(slide, 0.6, 1.3, 5.5, 3.5,
    "System: Classify this ASRS report.\nCategories:\n- Aircraft Equipment Problem\n- Airspace Violation\n- ATC Issue\n- Conflict\n- Deviation - Altitude\n- Deviation - Procedural\n- Deviation - Speed\n- Deviation - Track/Heading\n- Flight Deck/Cabin Event\n- Ground Event/Encounter\n- Ground Excursion\n- Ground Incursion\n- Inflight Event/Encounter\n\nReturn JSON list of matching categories.",
    "Basic Prompt");

  // Right: Taxonomy prompt
  addCodeBox(slide, 6.5, 1.3, 6.2, 3.5,
    "System: Classify using NASA ASRS taxonomy.\nCategories with subcategories:\n\n- Aircraft Equipment Problem\n  Less Severe | Critical\n  Hint: mechanical failures, system malfunctions\n\n- Conflict\n  NMAC | Airborne Conflict | Ground Conflict\n  Hint: TCAS RA, traffic proximity, near-miss\n\n- Deviation - Altitude\n  Overshoot | Undershoot | Crossing Restriction\n  Hint: assigned vs actual altitude, CFIT risk\n...\nReturn JSON list of matching categories.",
    "Taxonomy-Enriched Prompt");

  // Impact callouts
  const impY = 5.3;
  const impW = 3.6;
  const impGap = 0.3;
  const impX = (W - 3 * impW - 2 * impGap) / 2;

  addStatCallout(slide, impX, impY, impW, 1.3, "+0.040", "Macro-F1 подобрение\n(Qwen3-8B ZS)", C.green);
  addStatCallout(slide, impX + impW + impGap, impY, impW, 1.3, "+0.133", "Micro-F1 подобрение\n(Qwen3-8B ZS)", C.sky);
  addStatCallout(slide, impX + 2 * (impW + impGap), impY, impW, 1.3, "+0.073", "Macro-F1 подобрение\n(Qwen3-8B FS)", C.orange);
}

// ─── SLIDE 10: CLASSIC ML RESULTS (NEW) ─────────────────────────────────────

function slideClassicMLResults(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Резултати \u2014 Classic ML (TF-IDF + XGBoost)");

  // Left: bar chart
  const img = loadImage("classic_ml_f1_barchart.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.4, y: 1.1, w: 6.5, h: 5.0,
      sizing: { type: "contain", w: 6.5, h: 5.0 },
    });
  }

  // Right: headline metrics
  addStatCallout(slide, 7.4, 1.2, 2.5, 1.2, "0.691", "Macro-F1", C.green);
  addStatCallout(slide, 10.1, 1.2, 2.5, 1.2, "0.746", "Micro-F1", C.blue);

  slide.addText("Macro-AUC: 0.932", {
    x: 7.4, y: 2.55, w: 5.2, h: 0.35,
    fontFace: F.data, fontSize: 14, color: C.sky,
    bold: true, align: "center",
  });

  // Top 3 categories
  slide.addText("Най-добри категории:", {
    x: 7.4, y: 3.2, w: 5.2, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.text,
    bold: true,
  });
  const topCats = [
    "Aircraft Equipment Problem: F1 = 0.816",
    "Conflict: F1 = 0.801",
    "Deviation-Procedural: F1 = 0.795",
  ];
  topCats.forEach((c, i) => {
    slide.addText("\u2022  " + c, {
      x: 7.6, y: 3.6 + i * 0.35, w: 5.0, h: 0.32,
      fontFace: F.body, fontSize: 11, color: C.green,
    });
  });

  // Bottom 3 categories
  slide.addText("Най-слаби категории:", {
    x: 7.4, y: 4.8, w: 5.2, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.text,
    bold: true,
  });
  const botCats = [
    "Airspace Violation: F1 = 0.568",
    "Ground Excursion: F1 = 0.572",
    "Deviation-Speed: F1 = 0.577",
  ];
  botCats.forEach((c, i) => {
    slide.addText("\u2022  " + c, {
      x: 7.6, y: 5.2 + i * 0.35, w: 5.0, h: 0.32,
      fontFace: F.body, fontSize: 11, color: C.red,
    });
  });

  slide.addText("Редките категории са най-трудни \u2014 дисбаланс 30.3\u00d7 между най-честата и най-рядката", {
    x: 0.6, y: 6.7, w: W - 1.2, h: 0.35,
    fontFace: F.body, fontSize: 11, color: C.muted,
    italic: true, align: "center",
  });
}

// ─── SLIDE 11: ZERO-SHOT RESULTS (NEW) ──────────────────────────────────────

function slideZeroShotResults(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Резултати \u2014 Zero-Shot LLM");

  // Table: 6 ZS models ranked by Macro-F1
  const zsHeaders = ["Модел", "Промпт", "Macro-F1", "Micro-F1", "AUC"];
  const zsRows = [
    ["DeepSeek V3.2 + thinking", "taxonomy", "0.681", "0.723", "0.810"],
    ["Mistral Large 3", "taxonomy", "0.658", "0.712", "0.793"],
    ["DeepSeek V3.2", "taxonomy", "0.623", "0.693", "0.746"],
    ["Qwen3-8B", "taxonomy", "0.499", "0.605", "0.701"],
    ["Ministral 8B", "basic", "0.491", "0.543", "0.744"],
    ["Qwen3-8B", "basic", "0.459", "0.473", "0.727"],
  ];
  addTable(slide, zsHeaders, zsRows, {
    y: 1.15, colWidths: [3.5, 1.5, 1.5, 1.5, 1.5],
    rowHeight: 0.5, fontSize: 11, headerFontSize: 12,
  });

  // 3 insight boxes
  const insY = 4.8;
  const insW = 3.7;
  const insGap = 0.3;
  const insX = (W - 3 * insW - 2 * insGap) / 2;

  addDiagramBox(slide, insX, insY, insW, 1.7,
    "Мащаб е решаващ",
    "671\u2013675B MoE (0.658\u20130.681)\nзначително > 8B Dense (0.459\u20130.499).\nРазлика: +0.16\u20130.22 Macro-F1.",
    { accentColor: C.sky, titleSize: 12, lineSize: 10 });

  addDiagramBox(slide, insX + insW + insGap, insY, insW, 1.7,
    "Taxonomy помага",
    "Qwen3-8B: +0.040 Macro-F1,\n+0.133 Micro-F1 с taxonomy.\nПо-ясни категорийни граници\nза малки модели.",
    { accentColor: C.orange, titleSize: 12, lineSize: 10 });

  addDiagramBox(slide, insX + 2 * (insW + insGap), insY, insW, 1.7,
    "Thinking: +0.058 при 671B",
    "DeepSeek V3.2: 0.681 vs 0.623.\nНо 45\u00d7 по-бавно и 4.8\u00d7 по-скъпо.\nПри 8B: +0.007 (пренебрежимо).",
    { accentColor: C.red, titleSize: 12, lineSize: 10 });
}

// ─── SLIDE 12: FEW-SHOT + FINE-TUNING (NEW) ────────────────────────────────

function slideFewShotFineTuning(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Резултати \u2014 Few-Shot и Fine-Tuning");

  // Left: Few-Shot table
  slide.addText("Few-Shot LLM", {
    x: 0.6, y: 1.15, w: 5.5, h: 0.35,
    fontFace: F.heading, fontSize: 16, color: C.orange,
    bold: true,
  });

  const fsHeaders = ["Модел", "Промпт", "Macro-F1", "Micro-F1"];
  const fsRows = [
    ["Mistral Large 3", "taxonomy", "0.640", "0.686"],
    ["Ministral 8B", "basic", "0.540", "0.536"],
    ["Qwen3-8B + thinking", "taxonomy", "0.533", "0.556"],
    ["Qwen3-8B", "taxonomy", "0.526", "0.544"],
    ["Qwen3-8B", "basic", "0.453", "0.468"],
  ];
  addTable(slide, fsHeaders, fsRows, {
    x: 0.6, y: 1.55, colWidths: [2.5, 1.3, 1.2, 1.2],
    rowHeight: 0.45, fontSize: 10, headerFontSize: 11,
    centerX: false,
  });

  // Right: Fine-Tuning table
  slide.addText("Fine-Tuned LLM (QLoRA)", {
    x: 7.2, y: 1.15, w: 5.5, h: 0.35,
    fontFace: F.heading, fontSize: 16, color: C.red,
    bold: true,
  });

  const ftHeaders = ["Модел", "Macro-F1", "Micro-F1", "AUC"];
  const ftRows = [
    ["Qwen3-8B QLoRA", "0.510", "0.632", "0.700"],
    ["Ministral 8B LoRA/FP8", "0.489", "0.542", "0.744"],
  ];
  addTable(slide, ftHeaders, ftRows, {
    x: 7.2, y: 1.55, colWidths: [2.6, 1.2, 1.2, 1.1],
    rowHeight: 0.45, fontSize: 10, headerFontSize: 11,
    centerX: false,
  });

  // Key takeaways
  const takeaways = [
    { color: C.orange, title: "Few-shot: мащабът доминира", text: "Mistral Large 3 (675B) > всички 8B конфигурации. Few-shot на малък модел е по-слаб от zero-shot на голям." },
    { color: C.red, title: "Fine-tuning: значимо подобрение на Micro-F1", text: "Qwen3-8B QLoRA: +0.159 Micro-F1 vs zero-shot (0.632 vs 0.473). Но Macro-F1 остава по-нисък от zero-shot на 675B модел." },
    { color: C.sky, title: "Thinking mode: минимален ефект при 8B", text: "FS taxonomy + thinking: +0.007 Macro-F1 vs no-thinking. Не оправдава 6\u00d7 по-висока цена (A100 vs L4)." },
  ];

  const tkY = 4.1;
  const tkH = 0.85;
  takeaways.forEach((t, i) => {
    const ty = tkY + i * (tkH + 0.12);
    slide.addShape("rect", {
      x: 0.6, y: ty, w: W - 1.2, h: tkH,
      fill: { color: C.white },
      rectRadius: 0.08,
      shadow: makeShadow(),
    });
    slide.addShape("rect", {
      x: 0.6, y: ty, w: 0.06, h: tkH,
      fill: { color: t.color },
    });
    slide.addText(t.title, {
      x: 0.85, y: ty + 0.05, w: 3.5, h: 0.3,
      fontFace: F.heading, fontSize: 13, color: t.color,
      bold: true,
    });
    slide.addText(t.text, {
      x: 0.85, y: ty + 0.35, w: W - 1.7, h: tkH - 0.4,
      fontFace: F.body, fontSize: 11, color: C.text,
    });
  });
}

// ─── SLIDE 13: GRAND COMPARISON ─────────────────────────────────────────────

function slideGrandComparison(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Сравнение на всички модели (13 категории)");

  const img = loadImage("fig_grand_comparison.png");
  addFullFigure(slide, img, null, { y: 1.1, h: 4.2 });

  // Approach summary table
  const sumHeaders = ["Подход", "Най-добър модел", "Macro-F1", "Micro-F1"];
  const sumRows = [
    ["Classic ML", "TF-IDF + XGBoost", "0.691", "0.746"],
    ["Zero-Shot", "DeepSeek V3.2 + thinking", "0.681", "0.723"],
    ["Few-Shot", "Mistral Large 3", "0.640", "0.686"],
    ["Fine-Tuned", "Qwen3-8B QLoRA", "0.510", "0.632"],
  ];
  addTable(slide, sumHeaders, sumRows, {
    y: 5.5, colWidths: [2.0, 3.5, 1.5, 1.5],
    rowHeight: 0.45, fontSize: 11, headerFontSize: 12,
  });
}

// ─── SLIDE 14: CATEGORY HEATMAP ─────────────────────────────────────────────

function slideCategoryHeatmap(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "F1 по категория и модел (хийтмап)");

  const img = loadImage("fig_category_heatmap.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.3, y: 1.1, w: 8.8, h: 5.5,
      sizing: { type: "contain", w: 8.8, h: 5.5 },
    });
  }

  // Right side analysis
  addStatCallout(slide, 9.5, 1.3, 3.2, 1.2, "7 / 13", "категории \u2014\nXGBoost най-добър", C.green);
  addStatCallout(slide, 9.5, 2.8, 3.2, 1.2, "6 / 13", "категории \u2014\nDeepSeek V3.2 + thinking", C.sky);

  slide.addText("Семантичен анализ:", {
    x: 9.5, y: 4.3, w: 3.2, h: 0.3,
    fontFace: F.heading, fontSize: 13, color: C.text,
    bold: true,
  });
  slide.addText("XGBoost доминира при категории с ясни текстови маркери (ATC Issue, Ground Incursion).\n\nLLM печелят при категории изискващи разбиране на контекста (Conflict, Deviation-Altitude).", {
    x: 9.5, y: 4.65, w: 3.2, h: 2.0,
    fontFace: F.body, fontSize: 10, color: C.text,
    valign: "top", lineSpacingMultiple: 1.2,
  });
}

// ─── SLIDE 15: SCALE EFFECT ─────────────────────────────────────────────────

function slideScaleEffect(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Йерархия на факторите за LLM представяне");

  // Left — figure
  const img = loadImage("fig_scale_effect.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.4, y: 1.2, w: 6.5, h: 5.0,
      sizing: { type: "contain", w: 6.5, h: 5.0 },
    });
  }

  // Right — hierarchy
  const factors = [
    { num: "1", label: "Мащаб на модела", detail: "675B \u2192 0.658\u20130.681\n8B \u2192 0.459\u20130.510\n\u0394: +0.16\u20130.22 Macro-F1", color: C.sky },
    { num: "2", label: "Prompt Engineering", detail: "Taxonomy: +0.040\u20130.073\nFew-shot: +0.067 (ML3)\nТочни описания на границите", color: C.orange },
    { num: "3", label: "Fine-Tuning", detail: "QLoRA: +0.051 Macro-F1\n+0.159 Micro-F1\nНай-добър при малък модел", color: C.red },
    { num: "4", label: "Thinking Mode", detail: "671B: +0.058 Macro-F1\n8B: +0.007 (пренебрежимо)\n48 лейбъла: \u22120.003 (вреди)", color: C.muted },
  ];

  factors.forEach((f, i) => {
    const fy = 1.3 + i * 1.35;
    slide.addShape("rect", {
      x: 7.4, y: fy, w: 5.3, h: 1.15,
      fill: { color: C.white },
      rectRadius: 0.08,
      shadow: makeShadow(),
    });
    slide.addShape("rect", {
      x: 7.4, y: fy, w: 0.06, h: 1.15,
      fill: { color: f.color },
    });
    // Number circle
    slide.addShape("ellipse", {
      x: 7.6, y: fy + 0.15, w: 0.55, h: 0.55,
      fill: { color: f.color },
    });
    slide.addText(f.num, {
      x: 7.6, y: fy + 0.15, w: 0.55, h: 0.55,
      fontFace: F.heading, fontSize: 18, color: C.white,
      bold: true, align: "center", valign: "middle",
    });
    slide.addText(f.label, {
      x: 8.3, y: fy + 0.08, w: 3.0, h: 0.35,
      fontFace: F.heading, fontSize: 13, color: f.color,
      bold: true,
    });
    slide.addText(f.detail, {
      x: 8.3, y: fy + 0.4, w: 4.2, h: 0.7,
      fontFace: F.body, fontSize: 10, color: C.text,
      valign: "top",
    });
  });
}

// ─── SLIDE 16: SUBCATEGORY RESULTS (NEW) ────────────────────────────────────

function slideSubcategoryResults(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Подкатегории \u2014 48-лейбъл класификация");

  // Left: figure
  const img = loadImage("fig_parent_vs_sub.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.4, y: 1.1, w: 6.3, h: 4.5,
      sizing: { type: "contain", w: 6.3, h: 4.5 },
    });
  }

  // Right: comparison table
  const subHeaders = ["Модел", "Parent (13)", "Subcat (48)", "\u0394"];
  const subRows = [
    ["Classic ML", "0.691", "0.510", "\u22120.181"],
    ["Mistral Large 3", "0.658", "0.449", "\u22120.209"],
    ["DeepSeek V3.2", "0.623", "0.422", "\u22120.201"],
    ["Qwen3-8B", "0.499", "0.235", "\u22120.264"],
  ];
  addTable(slide, subHeaders, subRows, {
    x: 7.1, y: 1.2, colWidths: [2.1, 1.4, 1.4, 1.2],
    rowHeight: 0.5, fontSize: 11, headerFontSize: 11,
    centerX: false,
  });

  // Key findings
  slide.addText("Ключови наблюдения:", {
    x: 7.1, y: 3.8, w: 5.6, h: 0.3,
    fontFace: F.heading, fontSize: 13, color: C.text,
    bold: true,
  });
  const subFindings = [
    "Classic ML запазва AUC: 0.934 vs 0.932 (parent)",
    "Най-трудни: Ground Event подкатегории (F1 = 0.000\u20130.164)",
    "Най-добри: Haz Mat Violation (0.824), Smoke/Fire (0.815)",
    "LLM деградация е по-голяма (\u22120.264) от Classic ML (\u22120.181)",
    "Mistral Large 3 бие Classic ML на 11/48 подкатегории",
  ];
  subFindings.forEach((f, i) => {
    slide.addText("\u2022  " + f, {
      x: 7.3, y: 4.15 + i * 0.4, w: 5.4, h: 0.38,
      fontFace: F.body, fontSize: 10, color: C.text,
    });
  });

  slide.addText("48-лейбъл задачата е значително по-трудна: Macro-F1 пада с 0.18\u20130.26 за всички модели", {
    x: 0.6, y: 6.7, w: W - 1.2, h: 0.35,
    fontFace: F.body, fontSize: 11, color: C.muted,
    italic: true, align: "center",
  });
}

// ─── SLIDE 17: COST VS PERFORMANCE ──────────────────────────────────────────

function slideCostPerformance(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Цена срещу представяне");

  // Left: figure
  const img = loadImage("fig_cost_vs_performance.png");
  if (img) {
    slide.addImage({
      data: img,
      x: 0.3, y: 1.1, w: 7.0, h: 5.0,
      sizing: { type: "contain", w: 7.0, h: 5.0 },
    });
  }

  // Right: full cost table
  const costHeaders = ["Подход", "Цена", "Време", "Macro-F1"];
  const costRows = [
    ["Classic ML (XGBoost)", "$0", "<1 min", "0.691"],
    ["Mistral Large 3 ZS", "$0", "~5 min", "0.658"],
    ["DeepSeek V3.2 ZS", "$1.39", "6.5 min", "0.623"],
    ["DeepSeek V3.2 + thinking", "$6.73", "~5 hr", "0.681"],
    ["Qwen3-8B (all)", "~$19", "various", "0.510"],
  ];
  addTable(slide, costHeaders, costRows, {
    x: 7.6, y: 1.2, colWidths: [2.2, 0.9, 1.1, 1.2],
    rowHeight: 0.48, fontSize: 10, headerFontSize: 11,
    centerX: false,
  });

  // Total cost callout
  addStatCallout(slide, 7.6, 3.8, 5.4, 1.3, "~$53", "Обща цена на 22 експеримента\n(Modal: $38 + DeepInfra: $15 + Mistral: $0)", C.sky);

  slide.addText("Classic ML е едновременно най-евтин, най-бърз, и най-точен за 13-лейбъл задачата", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.text,
    italic: true, align: "center",
  });
}

// ─── SLIDE 18: WHY CLASSIC ML WINS (NEW) ────────────────────────────────────

function slideWhyClassicMLWins(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Защо Classic ML печели?");

  const cardW = 5.8;
  const cardH = 2.5;
  const gap = 0.35;
  const startX = (W - 2 * cardW - gap) / 2;
  const startY = 1.2;

  addCard(slide, startX, startY, cardW, cardH,
    "Domain-Specific N-грами",
    "TF-IDF улавя авиационни термини:\n\"runway incursion\", \"TCAS RA\", \"altitude deviation\".\nТези биграми директно кореспондират с\nимената на категориите \u2014 силен сигнал.",
    C.green);

  addCard(slide, startX + cardW + gap, startY, cardW, cardH,
    "Per-Label оптимизация",
    "13 независими класификатора, всеки с\nсобствен праг и scale_pos_weight.\nLLM правят една обща предикция за\nвсички категории наведнъж.",
    C.blue);

  addCard(slide, startX, startY + cardH + gap, cardW, cardH,
    "Достатъчно данни за BoW",
    "31,850 тренировъчни примера са достатъчни\nза bag-of-words подход. LLM от 8B нямат\nдостатъчно капацитет, а 675B не могат\nда се fine-tune (само API достъп).",
    C.orange);

  addCard(slide, startX + cardW + gap, startY + cardH + gap, cardW, cardH,
    "Градирани вероятности",
    "XGBoost дава вероятности (0\u20131) за всяка\nкатегория. LLM дават бинарно да/не с\nОптималният праг може да се настрои.\nAUC: 0.932 (Classic ML) vs 0.810 (DeepSeek).",
    C.red);

  slide.addText("Класическото ML не е \"остаряло\" \u2014 за структурирани задачи с ясни текстови маркери, то остава оптималният избор", {
    x: 0, y: 6.8, w: W, h: 0.35,
    fontFace: F.body, fontSize: 12, color: C.text,
    italic: true, align: "center",
  });
}

// ─── SLIDE 19: KEY FINDINGS ─────────────────────────────────────────────────

function slideKeyFindings(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.bg };
  addSlideHeader(slide, "Ключови находки");

  const findings = [
    { color: C.green, title: "Classic ML доминира", text: "TF-IDF + XGBoost (Macro-F1 0.691) превъзхожда всички LLM. Без GPU, <1 мин inference, $0 цена." },
    { color: C.sky, title: "Размерът на модела е решаващ", text: "675B MoE (0.658\u20130.681) >> 8B Dense (0.459\u20130.510). Fine-tuning на 8B не компенсира разликата." },
    { color: C.orange, title: "Taxonomy промптинг помага", text: "Подкатегории + подсказки: +0.040 Macro-F1, +0.133 Micro-F1 (Qwen3-8B ZS). Особено за малки модели." },
    { color: C.red, title: "Thinking mode \u2014 нюансиран ефект", text: "671B: +0.058 за 13 лейбъла, но \u22120.003 за 48 лейбъла (21.6% parse failures). 8B: +0.007." },
    { color: C.steel, title: "48-лейбъл задачата е значително по-трудна", text: "Macro-F1 пада с 0.18\u20130.26 за всички модели. Classic ML запазва AUC (0.934), LLM деградират повече." },
  ];

  const cardW = W - 1.2;
  const cardH = 0.82;
  const startY = 1.15;
  const gap = 0.1;

  findings.forEach((f, i) => {
    const cy = startY + i * (cardH + gap);
    slide.addShape("rect", {
      x: 0.6, y: cy, w: cardW, h: cardH,
      fill: { color: C.white },
      rectRadius: 0.08,
      shadow: makeShadow(),
    });
    slide.addShape("rect", {
      x: 0.6, y: cy, w: 0.06, h: cardH,
      fill: { color: f.color },
    });
    slide.addText(f.title, {
      x: 0.85, y: cy + 0.04, w: 3.5, h: 0.3,
      fontFace: F.heading, fontSize: 13, color: f.color,
      bold: true,
    });
    slide.addText(f.text, {
      x: 0.85, y: cy + 0.32, w: cardW - 0.5, h: cardH - 0.36,
      fontFace: F.body, fontSize: 11, color: C.text,
      valign: "top",
    });
  });

  // Practical recommendation box
  const recY = startY + 5 * (cardH + gap) + 0.1;
  slide.addShape("rect", {
    x: 0.6, y: recY, w: cardW, h: 0.75,
    fill: { color: C.greenBg },
    rectRadius: 0.08,
  });
  slide.addText("\u2705  Препоръка: За production класификация на ASRS доклади \u2014 TF-IDF + XGBoost с per-label оптимизация. LLM са полезни за exploration и когато няма обучителни данни.", {
    x: 0.8, y: recY + 0.05, w: cardW - 0.4, h: 0.65,
    fontFace: F.body, fontSize: 12, color: C.text,
    valign: "middle",
  });
}

// ─── SLIDE 20: CONCLUSION ───────────────────────────────────────────────────

function slideConclusion(pptx) {
  const slide = pptx.addSlide();
  slide.background = { fill: C.navy };

  // Title
  slide.addText("Заключение", {
    x: 0, y: 0.5, w: W, h: 0.55,
    fontFace: F.heading, fontSize: 30, color: C.white,
    bold: true, align: "center",
  });

  // Divider
  slide.addShape("rect", {
    x: W / 2 - 2, y: 1.2, w: 4, h: 0.03,
    fill: { color: C.sky },
  });

  // Main statement
  slide.addText("За мултилейбълна класификация на авиационни доклади,\nкласическото ML (TF-IDF + XGBoost) остава най-ефективният подход \u2014\nнай-висок F1, нулева цена за inference, и най-бърз.", {
    x: 1, y: 1.5, w: W - 2, h: 1.1,
    fontFace: F.body, fontSize: 17, color: C.white,
    align: "center", lineSpacingMultiple: 1.3,
  });

  // 3 Contributions
  slide.addText("Приноси", {
    x: 0, y: 2.8, w: W, h: 0.4,
    fontFace: F.heading, fontSize: 18, color: C.sky,
    bold: true, align: "center",
  });

  const contribs = [
    "Систематично сравнение на 4 подхода в 22 експеримента (zero-shot, few-shot, fine-tuned, Classic ML)",
    "Taxonomy-enriched prompting методология с измеримо подобрение (+0.040\u20130.073 Macro-F1)",
    "Пълен open-source codebase с възпроизводими резултати ($53 бюджет)",
  ];
  contribs.forEach((c, i) => {
    slide.addText(`${i + 1}.  ${c}`, {
      x: 1.5, y: 3.3 + i * 0.45, w: W - 3, h: 0.42,
      fontFace: F.body, fontSize: 13, color: C.white,
    });
  });

  // Future work
  slide.addText("Бъдещи насоки", {
    x: 0, y: 4.8, w: W, h: 0.4,
    fontFace: F.heading, fontSize: 18, color: C.sky,
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
      x: 2.5, y: 5.3 + i * 0.38, w: W - 5, h: 0.36,
      fontFace: F.body, fontSize: 13, color: C.white,
    });
  });

  // Divider
  slide.addShape("rect", {
    x: W / 2 - 2, y: 6.85, w: 4, h: 0.03,
    fill: { color: C.sky },
  });

  // Thank you + credits
  slide.addText("Благодаря за вниманието!", {
    x: 0, y: 6.2, w: W / 2, h: 0.5,
    fontFace: F.heading, fontSize: 20, color: C.sky,
    bold: true, align: "center",
  });
  slide.addText("github.com/rashevzarko-crypto/NBU-ASRS", {
    x: W / 2, y: 6.2, w: W / 2, h: 0.5,
    fontFace: F.data, fontSize: 13, color: C.muted,
    align: "center", valign: "middle",
  });

  slide.addText("Зарко Рашев  |  Ф№ F98363  |  НБУ  |  2026  |  Бюджет: ~$53", {
    x: 0, y: 7.0, w: W, h: 0.3,
    fontFace: F.body, fontSize: 12, color: C.muted,
    align: "center",
  });
}

// ─── SLIDE NUMBERS ──────────────────────────────────────────────────────────

function addSlideNumbers(pptx) {
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

// ─── MAIN ───────────────────────────────────────────────────────────────────

async function main() {
  console.log("Generating defense_presentation.pptx (20 slides)...");

  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_16x9";
  pptx.author = "Zharko Rashev";
  pptx.title = "Multi-Label Classification of Aviation Safety Reports";
  pptx.subject = "NBU Master's Thesis Defense";

  // Build all 20 slides
  slideTitle(pptx);              // 1  — Title (dark)
  slideResearchQuestion(pptx);   // 2  — Research Question
  slideDataset(pptx);            // 3  — Dataset
  slideMultiLabelExplained(pptx);// 4  — Multi-Label Explained (NEW)
  slideFourApproaches(pptx);     // 5  — Four Approaches
  slideClassicMLPipeline(pptx);  // 6  — Classic ML Pipeline (NEW)
  slideLLMArchitecture(pptx);    // 7  — LLM Architecture (NEW)
  slideQLoRA(pptx);              // 8  — QLoRA Fine-Tuning (NEW)
  slidePromptEngineering(pptx);  // 9  — Prompt Engineering (NEW)
  slideClassicMLResults(pptx);   // 10 — Classic ML Results (NEW)
  slideZeroShotResults(pptx);    // 11 — Zero-Shot Results (NEW)
  slideFewShotFineTuning(pptx);  // 12 — Few-Shot + Fine-Tuning (NEW)
  slideGrandComparison(pptx);    // 13 — Grand Comparison
  slideCategoryHeatmap(pptx);    // 14 — Category Heatmap
  slideScaleEffect(pptx);        // 15 — Scale Effect
  slideSubcategoryResults(pptx); // 16 — Subcategory Results (NEW)
  slideCostPerformance(pptx);    // 17 — Cost vs Performance
  slideWhyClassicMLWins(pptx);   // 18 — Why Classic ML Wins (NEW)
  slideKeyFindings(pptx);        // 19 — Key Findings
  slideConclusion(pptx);         // 20 — Conclusion (dark)

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
