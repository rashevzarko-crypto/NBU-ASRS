/**
 * NBU Master's Thesis Generator — Multi-Label Classification of ASRS Aviation Reports
 * Generates thesis.docx with all chapters, tables, figures, and bibliography in Bulgarian.
 *
 * Usage: node scripts/generate_thesis_docx.js
 * Output: thesis.docx in project root
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, HeadingLevel, PageBreak, BorderStyle,
  TableOfContents, ImageRun, Header, Footer, PageNumber, NumberFormat,
  ShadingType, VerticalAlign, Tab, TabStopType, TabStopPosition,
  convertInchesToTwip, ExternalHyperlink, SectionType, LineRuleType,
} = require("docx");

// ─── CONSTANTS ───────────────────────────────────────────────────────────────
const ROOT = path.resolve(__dirname, "..");
const RESULTS = path.join(ROOT, "results");
const FONT = "Times New Roman";
const FONT_SIZE_PT = 12;       // body text
const FONT_SIZE_HF = 24;       // half-points for docx (12pt = 24hp)
const HEADING1_SIZE = 28;       // 14pt
const HEADING2_SIZE = 26;       // 13pt
const HEADING3_SIZE = 24;       // 12pt
const LINE_SPACING = 360;      // 1.5 lines in twips (240 * 1.5)
const FIRST_LINE_INDENT = 720; // 1.27 cm ≈ 720 twips (standard Bulgarian indent)

// Margins: 2.5cm left, 2cm right/top/bottom
const CM_TO_TWIP = 567; // 1 cm ≈ 567 twips
const MARGINS = {
  top: 2 * CM_TO_TWIP,
  bottom: 2 * CM_TO_TWIP,
  left: 2.5 * CM_TO_TWIP,
  right: 2 * CM_TO_TWIP,
};

let figureCounter = 0;
let tableCounter = 0;

// ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

function nextFigure() { return ++figureCounter; }
function nextTable() { return ++tableCounter; }

/** Body paragraph with first-line indent and 1.5 spacing, justified */
function bodyParagraph(text, opts = {}) {
  const runs = typeof text === "string"
    ? [new TextRun({ text, font: FONT, size: FONT_SIZE_HF })]
    : text; // allow array of TextRun
  return new Paragraph({
    children: runs,
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: LINE_SPACING, before: 0, after: 0 },
    indent: opts.noIndent ? undefined : { firstLine: FIRST_LINE_INDENT },
    ...opts.extra,
  });
}

/** Paragraph without first-line indent */
function bodyNoIndent(text, opts = {}) {
  return bodyParagraph(text, { ...opts, noIndent: true });
}

/** Bold text run */
function bold(text) {
  return new TextRun({ text, font: FONT, size: FONT_SIZE_HF, bold: true });
}

/** Regular text run */
function run(text) {
  return new TextRun({ text, font: FONT, size: FONT_SIZE_HF });
}

/** Italic text run */
function italic(text) {
  return new TextRun({ text, font: FONT, size: FONT_SIZE_HF, italics: true });
}

/** Bold italic text run */
function boldItalic(text) {
  return new TextRun({ text, font: FONT, size: FONT_SIZE_HF, bold: true, italics: true });
}

/** English term in italic */
function en(text) {
  return new TextRun({ text, font: FONT, size: FONT_SIZE_HF, italics: true });
}

/** Heading1 paragraph — triggers page break */
function heading1(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: HEADING1_SIZE, bold: true })],
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120, line: LINE_SPACING },
    pageBreakBefore: true,
  });
}

/** Heading2 */
function heading2(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: HEADING2_SIZE, bold: true })],
    heading: HeadingLevel.HEADING_2,
    alignment: AlignmentType.LEFT,
    spacing: { before: 240, after: 120, line: LINE_SPACING },
  });
}

/** Heading3 */
function heading3(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: HEADING3_SIZE, bold: true })],
    heading: HeadingLevel.HEADING_3,
    alignment: AlignmentType.LEFT,
    spacing: { before: 200, after: 100, line: LINE_SPACING },
  });
}

/** Empty line */
function emptyLine() {
  return new Paragraph({
    children: [new TextRun({ text: "", font: FONT, size: FONT_SIZE_HF })],
    spacing: { line: LINE_SPACING },
  });
}

/** Caption paragraph (centered, italic, smaller) */
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 120, line: LINE_SPACING },
  });
}

/** Page break paragraph */
function pgBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

/** Load a PNG image and return ImageRun */
function loadImage(filename, widthInches = 5.5, heightInches = 3.5) {
  const imgPath = path.join(RESULTS, filename);
  if (!fs.existsSync(imgPath)) {
    console.warn(`WARNING: Image not found: ${imgPath}`);
    return null;
  }
  const data = fs.readFileSync(imgPath);
  return new ImageRun({
    data,
    transformation: {
      width: convertInchesToTwip(widthInches) / 15, // EMU to px approx
      height: convertInchesToTwip(heightInches) / 15,
    },
    type: "png",
  });
}

/** Paragraph containing a centered image */
function imageParagraph(filename, widthInches = 5.5, heightInches = 3.5) {
  const imgPath = path.join(RESULTS, filename);
  if (!fs.existsSync(imgPath)) {
    console.warn(`WARNING: Image not found: ${imgPath}`);
    return new Paragraph({
      children: [new TextRun({ text: `[IMAGE NOT FOUND: ${filename}]`, font: FONT, size: FONT_SIZE_HF, color: "FF0000" })],
      alignment: AlignmentType.CENTER,
    });
  }
  const data = fs.readFileSync(imgPath);
  return new Paragraph({
    children: [
      new ImageRun({
        data,
        transformation: {
          width: Math.round(widthInches * 96),
          height: Math.round(heightInches * 96),
        },
      }),
    ],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
  });
}

/** Table cell with text, optional shading */
function tc(text, opts = {}) {
  const fontSize = opts.fontSize || 20; // 10pt default for tables
  return new TableCell({
    children: [
      new Paragraph({
        children: [
          new TextRun({
            text: String(text),
            font: FONT,
            size: fontSize,
            bold: !!opts.bold,
          }),
        ],
        alignment: opts.align || AlignmentType.CENTER,
        spacing: { before: 20, after: 20 },
      }),
    ],
    shading: opts.shading ? {
      type: ShadingType.SOLID,
      color: opts.shading,
    } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    width: opts.width ? { size: opts.width, type: WidthType.PERCENTAGE } : undefined,
  });
}

/** Header row cell (light blue) */
function hc(text, opts = {}) {
  return tc(text, { ...opts, bold: true, shading: "D6E4F0" });
}

/** Create a data table from headers and rows arrays */
function makeDataTable(headers, rows, opts = {}) {
  const headerRow = new TableRow({
    children: headers.map(h => hc(h, { align: AlignmentType.CENTER })),
    tableHeader: true,
  });
  const dataRows = rows.map(row =>
    new TableRow({
      children: row.map((cell, i) =>
        tc(cell, {
          align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          bold: typeof cell === "string" && (cell.startsWith("MACRO") || cell.startsWith("MICRO")),
        })
      ),
    })
  );
  return new Table({
    rows: [headerRow, ...dataRows],
    width: { size: 100, type: WidthType.PERCENTAGE },
  });
}

/** Load CSV metrics file → { headers, rows } */
function loadMetricsCsv(filename) {
  const fp = path.join(RESULTS, filename);
  if (!fs.existsSync(fp)) {
    console.warn(`WARNING: CSV not found: ${fp}`);
    return { headers: [], rows: [] };
  }
  const lines = fs.readFileSync(fp, "utf-8").trim().split("\n");
  const headers = lines[0].split(",");
  const rows = lines.slice(1).map(l => {
    const parts = l.split(",");
    return parts.map((v, i) => {
      if (i === 0) return v;
      const num = parseFloat(v);
      return isNaN(num) ? v : num.toFixed(4);
    });
  });
  return { headers, rows };
}

/** Count total chars in an array of paragraphs */
function countChars(elements) {
  let total = 0;
  for (const el of elements) {
    if (el instanceof Paragraph && el.root) {
      // Walk the tree to find text
      const walk = (node) => {
        if (node && node.root) {
          for (const child of node.root) {
            walk(child);
          }
        }
        if (node && node.options && node.options.text) {
          total += node.options.text.length;
        }
      };
      walk(el);
    }
  }
  return total;
}

// Simple char counter from raw strings used in content functions
let charCount = 0;
function t(text) { charCount += text.length; return text; }

// ─── TITLE PAGE ──────────────────────────────────────────────────────────────

function buildTitlePage() {
  return [
    emptyLine(), emptyLine(),
    new Paragraph({
      children: [new TextRun({ text: t("НОВ БЪЛГАРСКИ УНИВЕРСИТЕТ"), font: FONT, size: 28, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
    }),
    new Paragraph({
      children: [new TextRun({ text: t('Департамент \u201EИнформатика\u201C'), font: FONT, size: FONT_SIZE_HF })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
    }),
    new Paragraph({
      children: [new TextRun({ text: t('Магистърска програма \u201EИзкуствен интелект\u201C'), font: FONT, size: FONT_SIZE_HF })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    }),
    emptyLine(), emptyLine(), emptyLine(),
    new Paragraph({
      children: [new TextRun({ text: t("МАГИСТЪРСКА ТЕЗА"), font: FONT, size: 32, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    }),
    emptyLine(),
    new Paragraph({
      children: [new TextRun({ text: t("Многоетикетна класификация на доклади за авиационна безопасност чрез машинно обучение и големи езикови модели"), font: FONT, size: 28, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
    }),
    new Paragraph({
      children: [new TextRun({ text: t("Multi-Label Classification of Aviation Safety Reports Using Machine Learning and Large Language Models"), font: FONT, size: 24, italics: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 480 },
    }),
    emptyLine(), emptyLine(), emptyLine(), emptyLine(),
    new Paragraph({
      children: [
        new TextRun({ text: t("Дипломант: "), font: FONT, size: FONT_SIZE_HF }),
        new TextRun({ text: t("Жарко Рашев"), font: FONT, size: FONT_SIZE_HF, bold: true }),
      ],
      alignment: AlignmentType.RIGHT,
      spacing: { after: 60 },
    }),
    new Paragraph({
      children: [
        new TextRun({ text: t("Научен ръководител: "), font: FONT, size: FONT_SIZE_HF }),
        new TextRun({ text: t("доц. д-р [Име Фамилия]"), font: FONT, size: FONT_SIZE_HF, bold: true }),
      ],
      alignment: AlignmentType.RIGHT,
      spacing: { after: 240 },
    }),
    emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
    new Paragraph({
      children: [new TextRun({ text: t("София, 2026"), font: FONT, size: FONT_SIZE_HF })],
      alignment: AlignmentType.CENTER,
    }),
  ];
}

// ─── ANNOTATION ──────────────────────────────────────────────────────────────

function buildAnnotation() {
  return [
    pgBreak(),
    new Paragraph({
      children: [new TextRun({ text: t("АНОТАЦИЯ"), font: FONT, size: HEADING1_SIZE, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 200, line: LINE_SPACING },
    }),
    bodyParagraph(t("Настоящата магистърска теза изследва систематично четири подхода за автоматична многоетикетна класификация на 172 183 доклада от системата за авиационна безопасност ASRS (Aviation Safety Reporting System) на NASA. Класификационната задача обхваща 13 категории аномалии, като всеки доклад може да принадлежи едновременно към множество категории.")),
    bodyParagraph(t("Сравняват се: (1) класическо машинно обучение с TF-IDF и XGBoost; (2) нулево обучение (zero-shot) с големи езикови модели (LLM); (3) обучение с малко примери (few-shot); и (4) фино настройване (fine-tuning) чрез QLoRA. Експериментите обхващат модели от 8 милиарда до 675 милиарда параметъра, включително Qwen3-8B, Ministral 8B, Mistral Large 3 и DeepSeek V3.2.")),
    bodyParagraph(t("Резултатите показват, че класическото машинно обучение (Macro-F1 = 0.691) значително превъзхожда всички LLM подходи. Най-добрият LLM резултат е постигнат от DeepSeek V3.2 с режим на разсъждение (Macro-F1 = 0.681), следван от Mistral Large 3 в нулево обучение (Macro-F1 = 0.658). Фино настройването на 8B модел подобрява Micro-F1 с +0.159 спрямо нулевото обучение, но остава далеч под класическия подход.")),
    bodyParagraph(t("Ключовият извод е, че инвестицията в инженерство на промптове (prompt engineering) дава по-добро съотношение цена/ефективност от фино настройването, а класическият подход с домейн-специфични текстови признаци остава златен стандарт за структурирана многоетикетна класификация.")),
    emptyLine(),
    new Paragraph({
      children: [
        bold(t("Ключови думи: ")),
        run(t("многоетикетна класификация, авиационна безопасност, ASRS, големи езикови модели, TF-IDF, XGBoost, QLoRA, промпт инженерство, NLP")),
      ],
      spacing: { line: LINE_SPACING },
    }),
    emptyLine(),
    emptyLine(),
    new Paragraph({
      children: [new TextRun({ text: t("ABSTRACT"), font: FONT, size: HEADING2_SIZE, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 200, line: LINE_SPACING },
    }),
    bodyParagraph(t("This master's thesis presents a systematic comparison of four approaches to automatic multi-label classification of 172,183 reports from NASA's Aviation Safety Reporting System (ASRS) across 13 anomaly categories. The compared approaches include: (1) classic machine learning with TF-IDF and XGBoost; (2) zero-shot classification with large language models (LLMs); (3) few-shot classification; and (4) fine-tuning via QLoRA. The experiments span models ranging from 8 billion to 675 billion parameters, including Qwen3-8B, Ministral 8B, Mistral Large 3, and DeepSeek V3.2.")),
    bodyParagraph(t("Results demonstrate that classic machine learning (Macro-F1 = 0.691) significantly outperforms all LLM approaches. The best LLM result is achieved by DeepSeek V3.2 with chain-of-thought reasoning (Macro-F1 = 0.681), followed by Mistral Large 3 in zero-shot mode (Macro-F1 = 0.658). Fine-tuning an 8B model improves Micro-F1 by +0.159 compared to zero-shot, but remains well below the classic approach. Prompt engineering with taxonomy enrichment provides the best cost-effectiveness ratio, improving Micro-F1 by +0.133 at zero additional cost.")),
    emptyLine(),
    new Paragraph({
      children: [
        bold(t("Keywords: ")),
        run(t("multi-label classification, aviation safety, ASRS, large language models, TF-IDF, XGBoost, QLoRA, prompt engineering, NLP")),
      ],
      spacing: { line: LINE_SPACING },
    }),
  ];
}

// ─── TABLE OF CONTENTS ──────────────────────────────────────────────────────

function buildTOC() {
  return [
    pgBreak(),
    new Paragraph({
      children: [new TextRun({ text: t("СЪДЪРЖАНИЕ"), font: FONT, size: HEADING1_SIZE, bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 200, line: LINE_SPACING },
    }),
    new TableOfContents("Съдържание", {
      hyperlink: true,
      headingStyleRange: "1-3",
    }),
    new Paragraph({
      children: [new TextRun({ text: t("(Моля, обновете полето с Ctrl+A → F9 след отваряне в Word)"), font: FONT, size: 20, italics: true, color: "808080" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120 },
    }),
  ];
}

// ─── INTRODUCTION ────────────────────────────────────────────────────────────

function buildIntroduction() {
  const p = [];
  p.push(heading1(t("УВОД")));

  p.push(bodyParagraph(t("Авиационната безопасност е един от най-критичните аспекти на съвременния въздушен транспорт. С нарастването на глобалния въздушен трафик нараства и обемът на доклади за инциденти и аномалии, които трябва да бъдат анализирани, за да се предотвратят бъдещи произшествия. Системата за авиационна безопасност ASRS (Aviation Safety Reporting System), управлявана от NASA, е най-голямата доброволна система за докладване на инциденти в авиацията в света. От създаването си през 1976 г. тя е събрала над 2 милиона доклада, подадени от пилоти, контрольори, механици и други авиационни специалисти.")));

  p.push(bodyParagraph(t("Всеки доклад в ASRS съдържа свободен текст — наратив, в който репортерът описва инцидента със собствени думи. Тези наративи се класифицират ръчно от експерти в NASA по множество категории аномалии, като един доклад може да принадлежи едновременно към няколко категории. Този процес на ръчна многоетикетна класификация е времеемък, субективен и труден за мащабиране при нарастващия обем от доклади.")));

  p.push(bodyParagraph(t("Автоматизацията на класификацията на ASRS доклади представлява значим изследователски проблем поради няколко причини. Първо, задачата е многоетикетна — всеки доклад може да има от 1 до 9 категории аномалии. Второ, категориите са силно небалансирани — най-честата категория (Deviation - Procedural) се среща в 65.4% от докладите, докато най-рядката (Ground Excursion) — само в 2.2%. Трето, текстовете са специализирани — съдържат авиационни съкращения (acft, rwy, apch, flt) и домейн-специфична терминология, която общоцелевите езикови модели може да не разпознават коректно.")));

  p.push(bodyParagraph(t("Появата на големите езикови модели (Large Language Models, LLM) през последните години постави нови въпроси за текстовата класификация: могат ли LLM моделите, обучени на огромни текстови корпуси, да превъзхождат традиционните подходи за машинно обучение в специализирани домейни? Могат ли техники като нулево обучение (zero-shot), обучение с малко примери (few-shot) и фино настройване (fine-tuning) да заменят класическите пайплайни от извличане на признаци и класификатори?")));

  p.push(bodyParagraph([
    run(t("Целта на настоящата магистърска теза е да даде систематичен отговор на тези въпроси чрез контролирано експериментално сравнение на четири основни подхода: (1) класическо машинно обучение (")),
    en(t("TF-IDF + XGBoost")),
    run(t("); (2) нулево обучение с LLM (")),
    en(t("zero-shot")),
    run(t("); (3) обучение с малко примери (")),
    en(t("few-shot")),
    run(t("); и (4) фино настройване на LLM чрез ")),
    en(t("QLoRA")),
    run(t(". Всички четири подхода се оценяват върху един и същ замразен тестов набор от 8 044 ASRS доклада.")),
  ]));

  p.push(bodyParagraph(t("Експериментите обхващат модели от различен мащаб — от 8 милиарда параметъра (Qwen3-8B, Ministral 8B) до 675 милиарда параметъра (Mistral Large 3, DeepSeek V3.2). Изследват се също така ефектите от инженерство на промптове (prompt engineering), включително обогатяване с таксономия и режим на разсъждение (chain-of-thought), както и разширение на задачата към 48 подкатегории.")));

  p.push(bodyParagraph(t("Общо са проведени 22 експеримента, обхващащи 5 различни модела, 4 типа подходи, 2 нива на гранулярност (13 категории и 48 подкатегории) и множество варианти на промптове. Всички експерименти са оценени по единна методология с идентични метрики — Precision, Recall, F1-Score и ROC-AUC, изчислени както в Macro, така и в Micro усредняване.")));

  p.push(bodyParagraph(t("Важен контекст на изследването е стремителното развитие на LLM технологиите в периода 2023–2026 г. Моделите от фамилиите GPT, LLaMA, Mistral, Qwen и DeepSeek показват непрекъснато подобряващи се резултати в широк спектър от NLP задачи. Същевременно, съществуващата литература за класификация на авиационни текстове е доминирана от класически подходи или BERT-базирани модели, без систематично изследване на съвременните LLM. Настоящата работа запълва тази празнина с актуални данни от модели от 2025–2026 г.")));
  p.push(bodyParagraph(t("Практическата значимост на изследването се определя от нуждата на авиационната индустрия от надеждни инструменти за автоматична класификация на инциденти. Ръчната класификация изисква специализирана експертиза, отнема значително време и е обект на субективност — различни анализатори могат да класифицират един и същ доклад различно. Автоматизираната класификация може да осигури последователност, бързина и мащабируемост, като същевременно служи като втори мнение за валидиране на ръчните решения.")));
  p.push(bodyParagraph(t("Изследването е организирано, както следва. Глава 1 представя литературен преглед на класификацията на текстове, LLM моделите и свързаните работи. Глава 2 описва методологията — набора от данни, моделите и експерименталния дизайн. Глава 3 представя резултатите от всички експерименти. Глава 4 дискутира резултатите и ограниченията. Заключението обобщава ключовите изводи и насоки за бъдещо развитие.")));

  p.push(bodyParagraph(t("Приносите на настоящата работа включват: (1) най-обширното сравнение на Classic ML и LLM подходи за ASRS класификация до момента; (2) количествен анализ на ефекта от промпт инженерство, показващ, че таксономично обогатеният промпт подобрява Micro-F1 с +0.133; (3) демонстрация, че мащабът на модела (8B vs 675B) е по-важен от техниката (fine-tuning vs zero-shot); (4) практическа оценка на разходите за различните подходи, полезна за бъдещи изследователи.")));

  p.push(bodyParagraph(t("Методологическият подход на работата се отличава с няколко характеристики, осигуряващи надеждност на резултатите. Всички модели са оценени на един и същ замразен тестов набор от 8 044 доклада, което елиминира вариацията от различни тестови набори. Промптовете и примерите са идентични за сравнимите експерименти, минимизирайки влиянието на случайни фактори. Всеки експеримент е документиран с пълна конфигурация, време на работа и разходи.")));

  p.push(bodyParagraph(t("Технологичният стек на изследването комбинира няколко платформи. Локалната среда (VS Code с Jupyter на Windows 11, 16 GB RAM, без GPU) се използва за обработка на данни, класическо машинно обучение и визуализация. Облачната платформа Modal предоставя GPU ресурси (NVIDIA L4 за инференция, A100 за обучение) на достъпни цени. API услугите на Mistral (безплатен Batch API) и DeepInfra (DeepSeek V3.2) допълват инфраструктурата за големите модели.")));

  return p;
}

// ─── CHAPTER 1: LITERATURE REVIEW ───────────────────────────────────────────

function buildChapter1() {
  const p = [];
  p.push(heading1(t("ГЛАВА 1. ЛИТЕРАТУРЕН ПРЕГЛЕД")));

  // 1.1
  p.push(heading2(t("1.1. Авиационна безопасност и системата ASRS")));
  p.push(bodyParagraph(t("Системата за докладване на авиационна безопасност ASRS (Aviation Safety Reporting System) е създадена от NASA през 1976 г. като отговор на нуждата от неказнително събиране на данни за инциденти в авиацията. ASRS приема доброволни и конфиденциални доклади от пилоти, въздушни контрольори, летищен персонал и други участници в авиационната система. Системата деидентифицира докладите, за да защити репортерите, и ги съхранява в публично достъпна база данни, която към момента съдържа над 2 милиона записа.")));
  p.push(bodyParagraph(t("Всеки доклад съдържа свободен текст — наратив, описващ инцидента, както и структурирани полета за тип въздухоплавателно средство, фаза на полета, метеорологични условия и други. Аналитиците на NASA класифицират всеки доклад по множество категории аномалии, като системата поддържа 13 основни категории на най-високо ниво и 48 подкатегории. Тази класификация е многоетикетна — един доклад може да получи от 1 до 9 категории.")));
  p.push(bodyParagraph(t("ASRS базата данни е ценен ресурс за изследователи в областта на авиационната безопасност, тъй като предоставя голям обем реални данни за инциденти и нередности. Ръчната класификация обаче е бавен процес — всеки доклад изисква внимателно четене и експертна оценка. Автоматизацията на тази класификация може значително да ускори анализа и да подобри последователността на решенията.")));
  p.push(bodyParagraph(t("Важна характеристика на ASRS е нейната неказнителна природа — Федералната авиационна администрация (FAA) гарантира, че информацията от докладите не може да бъде използвана за дисциплинарни действия срещу репортерите. Това стимулира доброволното докладване и осигурява по-пълна картина на инцидентите в сравнение с задължителните системи за докладване. Всеки доклад преминава през процес на деидентификация, при който личните данни, имената на летища и конкретните дати се заменят с анонимни заместители (например ZZZ за имена на летища).")));
  p.push(bodyParagraph(t("Текстовете в ASRS имат специфични характеристики, които ги отличават от общоцелевия текст. Авиационният жаргон включва множество съкращения: acft (aircraft), rwy (runway), apch (approach), flt (flight), twr (tower), fl (flight level), TRACON (Terminal Radar Approach Control). Много доклади са написани в телеграфен стил, без пълни изречения, и съдържат технически описания на процедури, инструменти и манипулации. Тази специфичност е едновременно предизвикателство (за общоцелевите модели) и предимство (за модели, базирани на н-грами, които могат да улавят тези уникални фрази).")));
  p.push(bodyParagraph(t("Историческият контекст на ASRS е свързан с инцидента на TWA Flight 514 през 1974 г. — катастрофа, дължаща се на комуникационна грешка, която е можела да бъде предотвратена, ако информацията от предишни подобни инциденти е била систематично споделяна. В отговор на тази и подобни трагедии, NASA създава ASRS като механизъм за безопасно споделяне на опит между авиационните професионалисти. Успехът на системата стимулира създаването на подобни програми в други страни и индустрии.")));
  p.push(bodyParagraph(t("Обемът на данните в ASRS нараства постоянно — от няколко хиляди доклада годишно в първите десетилетия до над 100 000 доклада годишно в последните години. Този ръст увеличава натоварването на аналитиците и прави ръчната класификация все по-трудно мащабируема. Автоматичната класификация може да служи като инструмент за предварителен филтър — автоматично категоризиране на входящите доклади, което позволява на експертите да фокусират вниманието си върху необичайни или критични случаи.")));
  p.push(bodyParagraph(t("Таксономията на ASRS категориите на аномалии е йерархична: 13 категории на най-високо ниво и 48 подкатегории. Категориите обхващат широк спектър от инциденти — от технически проблеми с оборудването на въздухоплавателното средство (Aircraft Equipment Problem) до нарушения на процедурите (Deviation - Procedural), конфликти в трафика (Conflict) и наземни инциденти (Ground Event/Encounter, Ground Incursion, Ground Excursion). Разграничаването между някои категории е фино — например Ground Excursion (напускане на предвидената повърхност) и Ground Incursion (неоторизирано навлизане на повърхност) описват противоположни движения, но често се бъркат.")));

  // 1.2
  p.push(heading2(t("1.2. Класическо машинно обучение за текстова класификация")));
  p.push(bodyParagraph([
    run(t("Класификацията на текстове е фундаментална задача в обработката на естествен език (NLP). Традиционният подход включва две основни стъпки: (1) извличане на признаци от текста и (2) обучение на класификатор върху извлечените признаци. Един от най-широко използваните методи за извличане на текстови признаци е TF-IDF (")),
    en(t("Term Frequency — Inverse Document Frequency")),
    run(t("), предложен от Salton и Buckley [1]. TF-IDF претегля всяка дума по нейната честота в документа и обратната документна честота в корпуса, като по този начин подчертава думи, които са специфични за конкретен документ.")),
  ]));
  p.push(bodyParagraph([
    run(t("Математически, TF-IDF тежестта на термин t в документ d от корпус D се изчислява като произведение на два компонента: TF(t,d) — честотата на термина в документа, и IDF(t,D) = log(N / df(t)), където N е общият брой документи, а df(t) — броят документи, съдържащи термина. При субкосинусова нормализация (")),
    en(t("sublinear_tf=True")),
    run(t(") TF компонентът се заменя с 1 + log(TF), което намалява влиянието на много честите думи в отделен документ. Резултатната TF-IDF матрица е разредена (sparse) — при 50 000 признака и 31 850 документа по-малко от 1% от стойностите са ненулеви, което прави я подходяща за алгоритми, оптимизирани за разредени данни.")),
  ]));
  p.push(bodyParagraph([
    run(t("За класификационната стъпка съществуват множество алгоритми. В настоящата работа основният класификатор е XGBoost (")),
    en(t("Extreme Gradient Boosting")),
    run(t(") [2], който е усъвършенствана реализация на метода на градиентно усилване на дървета (")),
    en(t("gradient boosted decision trees")),
    run(t("). XGBoost изгражда ансамбъл от дървета за решение последователно, като всяко ново дърво се обучава да коригира грешките на предишните. Основните му предимства са: (1) ефективната L1 и L2 регуларизация, предотвратяваща преобучаване; (2) хистограмно базирано изграждане на дървета (")),
    en(t("tree_method=hist")),
    run(t("), което ускорява обучението при високомерни данни; (3) вградената поддръжка за разредени матрици; (4) параметърът ")),
    en(t("scale_pos_weight")),
    run(t(" за компенсация на класов дисбаланс. XGBoost е показал високи резултати в широк спектър от задачи, включително спечелване на множество състезания на Kaggle [2].")),
  ]));
  p.push(bodyParagraph(t("За многоетикетна класификация най-простият подход е Binary Relevance — обучаване на независим бинарен класификатор за всяка категория. При 13 категории аномалии в ASRS това означава 13 независими XGBoost модела, всеки от които предсказва вероятността за принадлежност на доклада към съответната категория. Въпреки своята простота, този подход е ефективен и широко използван [3].")));
  p.push(bodyParagraph(t("Други класически подходи включват Logistic Regression и Linear SVM (Support Vector Machine), които са линейни модели, подходящи за високомерни разредени представяния като TF-IDF. Logistic Regression моделира вероятността за всяка категория чрез логистична функция и е ефективен за линейно разделими данни. LinearSVC (Linear Support Vector Classifier) търси хиперравнина с максимален отстъп (margin) между класовете и е устойчив към високомерни данни. В експерименталната фаза на настоящата работа са сравнени XGBoost, Logistic Regression и LinearSVC, като XGBoost постига най-добри резултати по Macro-F1, потвърждавайки предимството на нелинейния ансамблов подход пред линейните модели за тази задача.")));
  p.push(bodyParagraph([
    run(t("Друг широко използван градиентен ансамблов алгоритъм е LightGBM [26], който използва техниките ")),
    en(t("Gradient-based One-Side Sampling")),
    run(t(" (GOSS) и ")),
    en(t("Exclusive Feature Bundling")),
    run(t(" (EFB) за допълнително ускоряване на обучението. В настоящата работа е избран XGBoost поради неговата по-широка екосистема и по-добра документация за многоетикетни задачи, но LightGBM вероятно би дал сравними резултати.")),
  ]));
  p.push(bodyParagraph([
    run(t("Предимствата на класическия ML подход за текстова класификация включват: (1) интерпретируемост — TF-IDF тежестите показват кои думи са най-информативни; (2) ефективност — обучението отнема минути вместо часове; (3) стабилност — резултатите са детерминистични и възпроизводими; (4) ниска цена — не се изисква GPU хардуер. Недостатъците са: (1) загуба на семантична информация — ")),
    en(t("bag-of-words")),
    run(t(" моделите не улавят реда на думите и контекста; (2) неспособност за разпознаване на нови, невиждани формулировки; (3) необходимост от размечени (labeled) тренировъчни данни.")),
  ]));
  p.push(bodyParagraph([
    run(t("В контекста на многоетикетната класификация, ")),
    en(t("Binary Relevance")),
    run(t(" подходът има предимството, че всеки класификатор може да бъде настроен независимо за своята категория — различни прагове на решение, различно балансиране на класовете, различни хиперпараметри. Това е особено важно при силно небалансирани данни като ASRS, където рядка категория като Ground Excursion (2.2%) изисква различна стратегия от честа категория като Deviation - Procedural (65.4%). Алтернативни подходи за многоетикетна класификация включват ")),
    en(t("Classifier Chains")),
    run(t(" [3], които моделират зависимостите между етикетите, и ")),
    en(t("Label Powerset")),
    run(t(", който третира всяка комбинация от етикети като отделен клас.")),
  ]));

  // 1.3
  p.push(heading2(t("1.3. Големи езикови модели")));
  p.push(bodyParagraph([
    run(t("Революцията в обработката на естествен език започва с архитектурата ")),
    en(t("Transformer")),
    run(t(", представена от Vaswani и сътр. [4] през 2017 г. Трансформерът използва механизъм на внимание (")),
    en(t("self-attention")),
    run(t("), който позволява на модела да обработва всички позиции в последователността едновременно, вместо последователно. Това преодолява ограниченията на рекурентните невронни мрежи (RNN) по отношение на дългосрочните зависимости и паралелността на изчисленията.")),
  ]));
  p.push(bodyParagraph([
    run(t("Серията GPT (")),
    en(t("Generative Pre-trained Transformer")),
    run(t(") на OpenAI демонстрира силата на предобучените езикови модели. GPT-3 [5] с 175 милиарда параметъра показва забележителни способности за нулево обучение (zero-shot) и обучение с малко примери (few-shot), при които моделът решава нови задачи само чрез инструкции или малко примери в промпта, без допълнително обучение.")),
  ]));
  p.push(bodyParagraph([
    run(t("Настройката с инструкции (")),
    en(t("instruction tuning")),
    run(t(") [6] е техника, при която предобучен езиков модел се допълнително обучава върху набор от задачи, формулирани като инструкции. Това значително подобрява способността на модела да следва инструкции и да генерализира към нови задачи. Комбинирано с обучение чрез подкрепление от човешка обратна връзка (RLHF) [7], тази техника е в основата на съвременните чат асистенти.")),
  ]));
  p.push(bodyParagraph([
    run(t("Архитектурата ")),
    en(t("Mixture of Experts")),
    run(t(" (MoE) [8] позволява създаването на модели с огромен брой параметри, като при всяко извикване се активира само малка част от тях. Например Mistral Large 3 има 675 милиарда общи параметъра, но активира само 41 милиарда при всяко извикване. Това позволява висока производителност при по-ниски изчислителни разходи в сравнение с плътни (dense) модели със сходен брой параметри.")),
  ]));
  p.push(bodyParagraph([
    run(t("Механизмът на внимание (")),
    en(t("self-attention")),
    run(t(") изчислява за всяка позиция в последователността претеглена сума от представянията на всички останали позиции. Тежестите се определят чрез скаларно произведение между Query и Key вектори, мащабирано с корен от размерността (√d_k), следвано от Softmax нормализация. Резултатът се умножава с Value векторите. Многоглавото внимание (")),
    en(t("multi-head attention")),
    run(t(") разделя представянето на h глави, всяка от които улавя различни аспекти на зависимостите. При типичен модел с 8B параметъра се използват 32 глави с размерност 128 всяка. Тази архитектура позволява на модела да обръща внимание едновременно на синтактични, семантични и дискурсивни връзки.")),
  ]));
  p.push(bodyParagraph([
    run(t("Позиционното кодиране (")),
    en(t("positional encoding")),
    run(t(") е необходимо, тъй като механизмът на внимание сам по себе си не различава реда на токените. Оригиналният Transformer използва синусоидни позиционни кодирания, но съвременните модели прилагат ротационни позиционни вграждания (RoPE — ")),
    en(t("Rotary Position Embedding")),
    run(t("), които кодират позицията като ротация на векторното пространство. RoPE позволява екстраполация към по-дълги последователности от тези, срещани при обучението, което е важно за обработката на дълги авиационни наративи.")),
  ]));
  p.push(bodyParagraph(t("Ключово предимство на LLM моделите е способността им за нулево обучение (zero-shot learning) — решаване на нови задачи без допълнително обучение, само чрез подходяща текстова инструкция (промпт). За текстова класификация това означава, че моделът може да класифицира документ в предварително дефинирани категории, без да е видял нито един размечен пример. Качеството на zero-shot класификацията зависи силно от формулировката на промпта и от това доколко предобучаващите данни на модела покриват целевия домейн.")));
  p.push(bodyParagraph([
    run(t("Обучението с малко примери (")),
    en(t("few-shot learning")),
    run(t(") [5] разширява zero-shot подхода чрез включване на малък брой примери в промпта. Тези примери служат като контекстуални подсказки, които помагат на модела да разбере формата на очаквания изход и специфичните характеристики на категориите. В контекста на многоетикетна класификация, few-shot примерите трябва внимателно да бъдат подбрани, за да покрият разнообразието от категории, без да надхвърлят контекстния прозорец на модела.")),
  ]));
  p.push(bodyParagraph([
    run(t("Контекстният прозорец (")),
    en(t("context window")),
    run(t(") е максималният брой токени, които моделът може да обработи в една заявка. За Qwen3-8B контекстният прозорец е до 32K токена, а за Mistral Large 3 — до 128K. При few-shot класификация контекстният бюджет трябва да се разпредели между системната инструкция, примерите и текста за класификация. Това изисква компромис между броя и дължината на примерите.")),
  ]));
  p.push(bodyParagraph([
    run(t("Токенизацията (")),
    en(t("tokenization")),
    run(t(") е процесът на разбиване на текста на подсловни единици (токени), които моделът обработва. Съвременните LLM използват подсловна токенизация (напр. ")),
    en(t("BPE — Byte Pair Encoding")),
    run(t("), при която честите фрази и думи се кодират с по-малко токени, а редки думи — с повече. За английски текст приблизителното съотношение е 1 дума ≈ 1.3 токена. За авиационния домейн, специализираните съкращения могат да бъдат кодирани неефективно, ако не са представени в обучаващия корпус на токенизатора.")),
  ]));
  p.push(bodyParagraph([
    run(t("Предобучаващите данни (")),
    en(t("pre-training data")),
    run(t(") на LLM моделите обхващат огромни текстови корпуси — Common Crawl, Wikipedia, книги, научни статии и програмен код. Дали авиационните доклади (и конкретно ASRS) са представени в тези корпуси е неизвестно за повечето модели, но е вероятно, тъй като ASRS базата данни е публично достъпна. Това означава, че zero-shot резултатите могат да се дължат частично на запаметена информация от предобучението, а не само на генерализация.")),
  ]));
  p.push(bodyParagraph([
    run(t("Температурата (")),
    en(t("temperature")),
    run(t(") е параметър, контролиращ стохастичността на генерирането. При temperature=0 моделът винаги избира най-вероятния следващ токен (детерминистичен режим), което е предпочитано за класификация, тъй като осигурява възпроизводимост. По-високи стойности (0.5–1.0) добавят разнообразие, което е полезно за креативни задачи, но нежелателно за класификация.")),
  ]));
  p.push(bodyParagraph([
    run(t("Разсъжденческите способности на LLM могат да бъдат усилени чрез техниката верига от мисли (")),
    en(t("Chain-of-Thought")),
    run(t(", CoT) [9], при която моделът генерира междинни стъпки на разсъждение преди да даде крайния отговор. Някои модели като Qwen3 и DeepSeek V3 имат вграден режим на разсъждение (")),
    en(t("thinking mode")),
    run(t("), при който автоматично генерират вътрешни разсъждения.")),
  ]));

  // 1.4
  p.push(heading2(t("1.4. Методи за фино настройване: LoRA и QLoRA")));
  p.push(bodyParagraph([
    run(t("Фино настройването на големи езикови модели изисква значителни изчислителни ресурси — пълно фино настройване на модел с 8 милиарда параметъра изисква поне 32 GB GPU памет дори в смесена прецизност (FP16). LoRA (")),
    en(t("Low-Rank Adaptation")),
    run(t(") [10] предлага елегантно решение: вместо да се обновяват всичките милиарди параметри на модела, LoRA добавя малки нискорангови матрици (адаптери) към избрани слоеве на трансформера и обучава само тях. При ранг r = 16 и два целеви модула (q_proj, v_proj), обучаваните параметри са по-малко от 1% от общите.")),
  ]));
  p.push(bodyParagraph([
    run(t("QLoRA (")),
    en(t("Quantized Low-Rank Adaptation")),
    run(t(") [11] разширява LoRA чрез комбиниране с 4-битова квантизация на базовия модел. Използвайки NF4 (Normal Float 4-bit) квантизация с двойна квантизация, QLoRA намалява изискванията за памет почти 4 пъти спрямо FP16 — модел с 8B параметъра може да се обучава на единична GPU с 24 GB памет. Базовият модел се зарежда в 4-битова прецизност, а изчисленията се извършват в BFloat16, което запазва качеството на обучение.")),
  ]));
  p.push(bodyParagraph([
    run(t("NF4 квантизацията е специално проектирана за тегла на невронни мрежи, които обикновено следват нормално разпределение. За разлика от равномерната 4-битова квантизация (INT4), NF4 разпределя 16-те нива неравномерно — по-плътно около нулата (където са повечето тегла) и по-рядко в краищата. Двойната квантизация (")),
    en(t("double quantization")),
    run(t(") допълнително компресира квантизационните константи, спестявайки ~0.4 бита на параметър. Комбинацията от NF4 + двойна квантизация намалява размера на 8B модел от ~16 GB (FP16) на ~4 GB, позволявайки зареждане на единична GPU с 24 GB памет с достатъчно място за LoRA адаптерите и активациите.")),
  ]));
  p.push(bodyParagraph(t("В настоящата работа QLoRA е приложен върху Qwen3-8B с конфигурация: ранг r = 16, alpha = 16, целеви модули q_proj и v_proj, dropout 0.05. Обучението е проведено с оптимизатор paged_adamw_8bit (8-битова версия на AdamW [31] с поддръжка за свопинг на оптимизаторното състояние в CPU памет), планировчик cosine с warmup 5%, скорост на обучение 2e-5 и ефективен размер на партида 16 (batch 4 \u00D7 gradient accumulation 4).")));
  p.push(bodyParagraph([
    run(t("Важен практически аспект е съвместимостта на модела с QLoRA. Не всички модели поддържат стандартен QLoRA workflow. Ministral 3-8B например е съхранен като мултимодален модел (")),
    en(t("Mistral3ForConditionalGeneration")),
    run(t(") с FP8 квантизация, което предотвратява правилното прилагане на 4-битова NF4 квантизация. В този случай LoRA адаптерите се обучават върху FP8 базов модел, което ограничава ефективността на фино настройването. Qwen3-8B, от друга страна, е чист текстов модел (")),
    en(t("CausalLM")),
    run(t("), който поддържа стандартен QLoRA без ограничения.")),
  ]));
  p.push(bodyParagraph([
    run(t("Инференцията на фино настроения модел се извършва чрез зареждане на базовия модел заедно с обучения LoRA адаптер. Библиотеката ")),
    en(t("vLLM")),
    run(t(" поддържа динамично зареждане на LoRA адаптери чрез параметрите ")),
    en(t("enable_lora=True")),
    run(t(" и ")),
    en(t("max_lora_rank=16")),
    run(t(". Това позволява ефективна инференция с батчинг, като базовият модел е зареден в паметта веднъж, а LoRA адаптерът добавя минимален допълнителен overhead.")),
  ]));

  // 1.5
  p.push(heading2(t("1.5. Многоетикетна класификация и метрики")));
  p.push(bodyParagraph([
    run(t("Многоетикетната класификация (")),
    en(t("multi-label classification")),
    run(t(") е задача, при която всеки обект може да принадлежи към множество класове едновременно, за разлика от многокласовата класификация, при която всеки обект принадлежи точно към един клас. При ASRS докладите медианата е 2 категории на доклад, а 78% от докладите имат повече от една категория.")),
  ]));
  p.push(bodyParagraph([
    run(t("За оценка на многоетикетни класификатори се използват разширения на стандартните метрики ")),
    en(t("Precision")),
    run(t(" (точност), ")),
    en(t("Recall")),
    run(t(" (пълнота) и ")),
    en(t("F1-Score")),
    run(t(" (хармонична средна). Двете основни схеми за агрегиране са:")),
  ]));
  p.push(bodyParagraph([
    bold(t("Macro-усредняване: ")),
    run(t("Изчислява метриката за всяка категория поотделно и ги усреднява. Третира всички категории еднакво, независимо от честотата им. Чувствителна е към резултатите по редки категории.")),
  ]));
  p.push(bodyParagraph([
    bold(t("Micro-усредняване: ")),
    run(t("Агрегира всички истински положителни, фалшиви положителни и фалшиви отрицателни стойности глобално и изчислява метриките. Дава по-голяма тежест на честите категории.")),
  ]));
  p.push(bodyParagraph([
    run(t("ROC-AUC (")),
    en(t("Receiver Operating Characteristic — Area Under Curve")),
    run(t(") [12] измерва способността на модела да разграничава положителни от отрицателни примери при различни прагове. AUC стойност 0.5 е равна на случайно предсказване, а 1.0 — на перфектно разделяне.")),
  ]));
  p.push(bodyParagraph([
    run(t("Стратифицираното семплиране в многоетикетен контекст е усложнено от факта, че трябва да се запазят пропорциите на всички комбинации от етикети. В настоящата работа се използва ")),
    en(t("MultilabelStratifiedShuffleSplit")),
    run(t(" от библиотеката ")),
    en(t("iterstrat")),
    run(t(" [13], който апроксимативно запазва разпределението на етикетите между обучаващия и тестовия набор.")),
  ]));
  p.push(bodyParagraph(t("В настоящата работа основната метрика за сравнение е Macro-F1, тъй като в контекста на авиационната безопасност всички категории аномалии са потенциално важни \u2014 рядка категория като Ground Excursion (напускане на пистата) може да бъде критично опасна. Micro-F1 е допълнителна метрика, отразяваща цялостната производителност, претеглена по честотата на категориите.")));
  p.push(bodyParagraph([
    run(t("Важно техническо уточнение: при многоетикетна класификация метриките трябва да се изчисляват върху двумерни матрици (n_samples \u00D7 n_labels), а не върху изравнени (flattened) вектори. Изравняването чрез ")),
    en(t(".ravel()")),
    run(t(" преди изчислението превръща задачата от многоетикетна в бинарна, което изкуствено повишава метриките. Тази грешка е идентифицирана и коригирана в ранна фаза на изследването.")),
  ]));
  p.push(bodyParagraph([
    run(t("За LLM моделите ROC-AUC се изчислява от бинарните предсказания (0 или 1), тъй като моделите не произвеждат градирани вероятности. Това е принципна разлика спрямо Classic ML, който генерира непрекъснати вероятности за всяка категория. Резултатът е, че AUC стойностите на LLM са ограничени отгоре \u2014 те зависят от точността и пълнотата при единствения наличен праг, докато Classic ML AUC измерва способността за ранжиране при всички възможни прагове.")),
  ]));

  // 1.6
  p.push(heading2(t("1.6. Свързани работи")));
  p.push(bodyParagraph(t("Класификацията на ASRS доклади е изследвана от множество автори. Ранните работи използват класически подходи: Bag-of-Words представяния с Naive Bayes и SVM класификатори [14]. По-нови изследвания прилагат дълбоко обучение — LSTM и CNN архитектури за извличане на текстови признаци [15]. Тези работи обаче обикновено се фокусират върху двукласова класификация или многокласова класификация (един етикет на доклад), а не върху многоетикетната задача.")));
  p.push(bodyParagraph(t("Kuhn [14] е сред първите автори, изследващи автоматизираната класификация на авиационни инцидентни доклади. Работата идентифицира основните предизвикателства: специализиран лексикон, вариативност в стила на описание, припокриващи се категории и необходимост от експертно знание за коректната класификация. Тези предизвикателства остават актуални и днес, две десетилетия по-късно, и мотивират изследванията с по-мощни модели.")));
  p.push(bodyParagraph(t("Robinson и сътр. [15] прилагат дълбоко обучение (LSTM и CNN) за класификация на типове авиационни инциденти и постигат резултати, сравними с по-простите подходи, но с по-висока изчислителна цена. Техният принос е в демонстрацията, че дълбокото обучение може да улови контекстуални зависимости в авиационния текст, но ограниченият обем на данните намалява предимството спрямо класическите методи.")));
  p.push(bodyParagraph([
    run(t("Zhang и сътр. [18] прилагат BERT (")),
    en(t("Bidirectional Encoder Representations from Transformers")),
    run(t(") [20] за класификация на ASRS доклади и постигат F1 резултати между 0.70 и 0.85 за различни типове инциденти. Техният подход се възползва от двупосочното внимание на BERT, което улавя контекста и от двете страни на всяка дума. Ограничението обаче е, че работят с многокласова (а не многоетикетна) формулировка, което опростява задачата.")),
  ]));
  p.push(bodyParagraph([
    run(t("Приложението на LLM за текстова класификация е активна изследователска област. Sun и сътр. [16] демонстрират, че GPT-3.5 и GPT-4 постигат конкурентни резултати в задачи за класификация на настроения и тематична класификация при нулево обучение. Wang и сътр. [17] изследват фино настройването на LLaMA с LoRA за задачи за класификация и намират, че този подход е ефективен при ограничен обем тренировъчни данни.")),
  ]));
  p.push(bodyParagraph(t("В областта на авиационната безопасност Zhang и сътр. [18] прилагат BERT за класификация на типове инциденти от ASRS доклади. Те постигат F1 резултати между 0.70 и 0.85 за различни типове инциденти, но работят с многокласова, а не многоетикетна формулировка. Rose и сътр. [19] изследват използването на LLM за извличане на информация от авиационни доклади, но не провеждат систематично сравнение между класически и LLM подходи.")));
  p.push(bodyParagraph(t("Настоящата работа допринася към литературата с няколко нови аспекта: (1) систематично четирипосочно сравнение на класически ML и три LLM подхода върху една и съща задача; (2) многоетикетна формулировка с 13 категории; (3) анализ на ефекта от мащаба на модела (от 8B до 675B параметъра); (4) изследване на ролята на промпт инженерството и режима на разсъждение; (5) разширение към 48 подкатегории.")));
  p.push(bodyParagraph(t("В по-широкия контекст на NLP, сравнението между класическите ML подходи и LLM моделите е обект на нарастващ изследователски интерес. Множество проучвания [20, 21] показват, че предобучените трансформери (BERT, GPT) превъзхождат традиционните модели при задачи с ограничени тренировъчни данни. Същевременно, при наличие на големи размечени набори данни и специализирани домейни, класическите подходи остават конкурентни [16]. Настоящата работа допринася към тази дискусия с конкретни данни от авиационния домейн.")));
  p.push(bodyParagraph([
    run(t("Еволюцията на предобучените езикови модели преминава през няколко ключови етапа. BERT [20] въвежда двупосочното предобучение и показва, че фино настройването на предобучен модел може да постигне най-съвременни резултати в множество NLP задачи. GPT серията [21] демонстрира, че еднопосочните авторегресивни модели са също толкова мощни, особено при мащабиране. LLaMA [22] от Meta отваря достъпа до висококачествени отворени модели, а Mistral [23] показва, че малки модели (7B) могат да бъдат конкурентни на по-големи чрез архитектурни оптимизации като ")),
    en(t("Sliding Window Attention")),
    run(t(" и ")),
    en(t("Grouped-Query Attention")),
    run(t(".")),
  ]));
  p.push(bodyParagraph(t("Sun и сътр. [16] провеждат систематично сравнение на LLM за текстова класификация и установяват, че резултатите зависят силно от формулировката на промпта и от специфичността на домейна. За общоцелеви задачи (класификация на настроения, тематична класификация) LLM постигат конкурентни резултати, но за специализирани домейни с технически лексикон — резултатите са по-ниски. Тези наблюдения са потвърдени от настоящата работа в авиационния контекст.")));
  p.push(bodyParagraph(t("Wang и сътр. [17] изследват фино настройването на LLaMA с LoRA за задачи за класификация и намират, че този подход е ефективен при ограничен обем тренировъчни данни. Техните резултати показват, че LoRA може да адаптира предобучен модел към целева задача с по-малко от 1% допълнителни параметри, запазвайки повечето от знанията от предобучението. Настоящата работа потвърждава тази находка с QLoRA на Qwen3-8B, макар и резултатите да остават под тези на класическия ML подход.")));
  p.push(bodyParagraph(t("Rose и сътр. [19] изследват използването на LLM за извличане на структурирана информация от авиационни доклади, включително тип на инцидента, засегнати системи и фази на полета. Техният подход се различава от класификацията по това, че цели извличане на конкретни полета, а не присвояване на категории. Въпреки различната задача, тяхната работа потвърждава потенциала на LLM за обработка на авиационни текстове.")));
  p.push(bodyParagraph(t("Многоетикетната класификация [27] е подобласт на машинното обучение с богата теоретична основа. Различните подходи за решаване на многоетикетни задачи — Binary Relevance, Classifier Chains, Label Powerset, RAKEL — имат различни компромиси между изчислителна сложност и моделиране на зависимостите между етикетите. В контекста на ASRS класификацията, Binary Relevance е достатъчно ефективен, тъй като XGBoost улавя нелинейните връзки между признаците, компенсирайки липсата на моделиране на междуетикетните зависимости.")));
  p.push(bodyParagraph(t("Библиотеката scikit-learn [29] предоставя стандартизирани имплементации на метриките и класификаторите, използвани в настоящата работа. За XGBoost е използвана официалната Python библиотека xgboost, а за стратифицираното семплиране — iterstrat. Стандартизацията на инструментите осигурява възпроизводимост на резултатите и съвместимост с други изследвания в областта.")));
  p.push(bodyParagraph(t("Промпт инженерството (prompt engineering) се утвърждава като отделна дисциплина в приложението на LLM. Изследвания показват, че формулировката на промпта може да промени резултата с десетки процентни пункта [9, 16]. Техники като включване на дефиниции на категориите, дискриминативни подсказки и йерархична таксономична информация са доказали ефективността си в различни класификационни задачи. Настоящата работа изследва специфично тези техники в контекста на авиационната безопасност.")));
  p.push(bodyParagraph([
    run(t("Архитектурата ")),
    en(t("Mixture of Experts")),
    run(t(" придобива особено значение за практическото приложение на LLM. Модели като Mistral Large 3 (675B параметъра, 41B активни) и DeepSeek V3.2 (671B параметъра) демонстрират, че е възможно да се постигнат резултати, сравними с плътни модели от същия мащаб, при значително по-ниски изчислителни разходи за инференция [23, 25]. Тази архитектура е ключова за достъпността на големите модели чрез API услуги.")),
  ]));

  // 1.7 Transfer Learning and Domain Adaptation
  p.push(heading2(t("1.7. Трансферно обучение и адаптация към домейна")));
  p.push(bodyParagraph([
    run(t("Трансферното обучение (")),
    en(t("transfer learning")),
    run(t(") е парадигма, при която модел, обучен за една задача, се адаптира за друга. В контекста на NLP, предобучените езикови модели (BERT, GPT, LLaMA, Qwen) представляват форма на трансферно обучение \u2014 знанията, придобити при предобучението върху огромни текстови корпуси, се прехвърлят към целевата задача чрез фино настройване или промпт-базирани подходи.")),
  ]));
  p.push(bodyParagraph([
    run(t("За авиационния домейн трансферното обучение е особено интересно, тъй като ASRS текстовете се различават значително от общоцелевия текст в предобучаващите корпуси. Авиационният жаргон, телеграфният стил на писане и домейн-специфичните категории създават ")),
    en(t("domain shift")),
    run(t(" \u2014 разлика между разпределението на текстовете в предобучаващия корпус и целевия домейн. Ефективността на трансферното обучение зависи от степента на този domain shift.")),
  ]));
  p.push(bodyParagraph(t("Zero-shot класификацията с LLM представлява най-екстремната форма на трансферно обучение \u2014 моделът се прилага директно към целевата задача без никаква адаптация, разчитайки изцяло на знанията от предобучението. Few-shot подходът добавя минимална адаптация чрез примери в промпта. Fine-tuning (QLoRA) адаптира параметрите на модела, но само малка част от тях (~0.5% при ранг 16). Classic ML, от друга страна, не използва трансферно обучение \u2014 моделът се обучава изцяло от нулата върху целевите данни.")));
  p.push(bodyParagraph(t("Резултатите от настоящата работа показват, че трансферното обучение (zero-shot/few-shot) е по-ефективно при по-големи модели, които имат по-широко покритие на предобучаващия корпус. Малките модели (8B) нямат достатъчно капацитет за авиационния домейн и показват значителен domain shift. Големите модели (671-675B) вероятно са срещали авиационни текстове (и може би ASRS данни) по време на предобучението, което намалява domain shift и подобрява zero-shot резултатите.")));
  p.push(bodyParagraph([
    run(t("Интересен аспект на адаптацията към домейна е ролята на промпт инженерството като форма на ")),
    en(t("test-time adaptation")),
    run(t(" \u2014 адаптация по време на инференция, без промяна на параметрите на модела. Таксономичният промпт може да се разглежда като вид мек (soft) трансфер на експертно знание за авиационната класификация, кодирано в текстов формат. Ефективността на тази техника (Micro-F1 +0.133 за Qwen3-8B) показва, че дори без промяна на параметрите, значително подобрение е възможно чрез добре конструирани инструкции.")),
  ]));
  p.push(bodyParagraph(t("В по-широк контекст, трансферното обучение чрез LLM може да се разглежда като компромис между гъвкавост и специализация. LLM моделите са гъвкави \u2014 могат да класифицират без размечени данни, да променят категориите само чрез промяна на промпта и да обработват нови видове текстове. Classic ML е специализиран \u2014 оптимизиран за конкретната задача и набор от данни, но изисква пълна преработка при промяна на категориите или домейна. Изборът между тези подходи зависи от конкретните изисквания: наличие на размечени данни, стабилност на категориите, нужда от бърза адаптация и бюджетни ограничения.")));

  return p;
}

// ─── CHAPTER 2: METHODOLOGY ─────────────────────────────────────────────────

function buildChapter2() {
  const p = [];
  p.push(heading1(t("ГЛАВА 2. МЕТОДОЛОГИЯ")));

  // 2.1
  p.push(heading2(t("2.1. Набор от данни")));
  p.push(bodyParagraph(t("Източникът на данни е публичната база данни на NASA ASRS, съдържаща доклади за авиационни инциденти, подадени доброволно от участници в авиационната система. Изтеглени са 61 CSV файла с общо 282 371 записа. След дедупликация по уникален номер ACN (Accession Number) остават 172 183 уникални доклада.")));
  p.push(bodyParagraph(t("Текстовото поле, използвано за класификация, е Report Narrative — свободен текст, описващ инцидента. При 15 720 доклада (9.1%) е наличен втори наратив, който е конкатениран към първия. Средната дължина на наратива е 265 думи (медиана 212), със стандартно отклонение 213 думи.")));
  p.push(bodyParagraph(t("Категориите аномалии са извлечени от полето Anomaly на ASRS, което съдържа 8 272 уникални низа. Тези низове са картографирани към 13 категории на най-високо ниво чрез стриктно съвпадение на префикси. Всичките 172 183 доклада имат поне една картографирана категория.")));

  // Category distribution table
  const tNum = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNum}. Разпределение на категориите аномалии (172 183 доклада)`))]));
  p.push(makeDataTable(
    ["Категория", "Брой", "% от докладите"],
    [
      ["Deviation - Procedural", "112 606", "65.4%"],
      ["Aircraft Equipment Problem", "49 305", "28.6%"],
      ["Conflict", "46 285", "26.9%"],
      ["Inflight Event/Encounter", "38 658", "22.5%"],
      ["ATC Issue", "29 422", "17.1%"],
      ["Deviation - Altitude", "28 369", "16.5%"],
      ["Deviation - Track/Heading", "20 268", "11.8%"],
      ["Ground Event/Encounter", "14 234", "8.3%"],
      ["Ground Incursion", "12 601", "7.3%"],
      ["Flight Deck/Cabin Event", "12 291", "7.1%"],
      ["Airspace Violation", "6 834", "4.0%"],
      ["Deviation - Speed", "5 000", "2.9%"],
      ["Ground Excursion", "3 718", "2.2%"],
    ],
  ));
  p.push(emptyLine());

  p.push(bodyParagraph(t("Данните показват значителен дисбаланс: Deviation - Procedural се среща в 65.4% от докладите, докато Ground Excursion — само в 2.2%, което дава съотношение на дисбаланс 30.3:1. Медианата на етикетите на доклад е 2, като 22% от докладите имат само една категория, а 78% — две или повече.")));

  // Label count distribution table
  const tNumLC = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumLC}. Разпределение на броя етикети на доклад (172 183 доклада)`))]));
  p.push(makeDataTable(
    ["Брой етикети", "Брой доклади", "% от общото"],
    [
      ["1 етикет", "38 403", "22.3%"],
      ["2 етикета", "77 602", "45.1%"],
      ["3 етикета", "41 494", "24.1%"],
      ["4 етикета", "12 224", "7.1%"],
      ["5 етикета", "2 178", "1.3%"],
      ["6 етикета", "260", "0.15%"],
      ["7 етикета", "21", "0.01%"],
      ["9 етикета", "1", "<0.01%"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Разпределението на броя етикети показва, че най-честият случай е 2 етикета на доклад (45.1%), следван от 3 етикета (24.1%). Доклади с 5 или повече етикета са изключително редки (1.5% от общото). Един единствен доклад има 9 етикета — максималният брой в набора. Тази статистика потвърждава многоетикетния характер на задачата и необходимостта от специализирани методи за многоетикетна класификация, вместо по-простите многокласови подходи.")));
  // Text statistics table
  const tNumTS = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumTS}. Статистика на текстовите наративи`))]));
  p.push(makeDataTable(
    ["Статистика", "Обучаващ набор", "Тестов набор", "Общо"],
    [
      ["Брой доклади", "31 850", "8 044", "39 894"],
      ["Средна дължина (думи)", "264.9", "266.9", "265.3"],
      ["Медиана (думи)", "212", "212", "212"],
      ["Мин. дължина (думи)", "2", "2", "2"],
      ["Макс. дължина (думи)", "3 657", "2 477", "3 657"],
      ["Средно знаци", "1 438", "1 449", "1 440"],
      ["Средно токени (×1.3)", "344", "347", "345"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Текстовата статистика показва, че наративите варират значително по дължина \u2014 от 2 думи до 3 657 думи. Средната дължина от 265 думи (~345 токена) е добре в рамките на контекстния прозорец на всички използвани модели. Нито един доклад няма празен наратив (0 null/empty), което осигурява пълно покритие на данните.")));
  p.push(bodyParagraph(t("Най-честите думи (след премахване на стоп думите) отразяват специфичния авиационен лексикон: aircraft (52 957 появявания), ft (46 091), acft (43 716), rwy (42 498), time (30 200), runway (27 068), flight (24 787), apch (23 307). Съкращението zzz (17 296) е анонимизиращ заместител на ASRS за имена на летища и локации. Тези домейн-специфични термини и съкращения са ключови признаци за TF-IDF класификатора, но може да не бъдат разпознати от LLM модели, чийто токенизатор не е оптимизиран за авиационния лексикон.")));
  p.push(bodyParagraph(t("Анализът на най-честите комбинации от етикети разкрива, че Deviation - Procedural участва в 8 от 10-те най-чести комбинации. Най-честата двойка е Aircraft Equipment Problem + Deviation - Procedural (6.3% от извадката), следвана от Deviation - Altitude + Deviation - Procedural (5.6%). Тази доминация на процедурните отклонения отразява факта, че повечето инциденти включват нарушение на някаква процедура.")));

  // Co-occurrence heatmap
  const fNum = nextFigure();
  p.push(imageParagraph("co_occurrence_heatmap.png", 5.0, 4.5));
  p.push(caption(t(`Фигура ${fNum}. Матрица на съвместно появяване на категориите`)));

  p.push(bodyParagraph(t("Фигура " + fNum + " показва матрицата на съвместно появяване на категориите. Deviation - Procedural има високи стойности на съвместно появяване с почти всички останали категории, което отразява нейната широка дефиниция \u2014 повечето инциденти включват някакво процедурно отклонение.")));

  // Top label combinations table
  const tNumCombo = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumCombo}. Най-чести комбинации от категории`))]));
  p.push(makeDataTable(
    ["Ранг", "Комбинация", "Брой", "% от извадката"],
    [
      ["1", "Aircraft Equip. + Dev-Procedural", "2 495", "6.3%"],
      ["2", "Dev-Altitude + Dev-Procedural", "2 245", "5.6%"],
      ["3", "Aircraft Equipment Problem (само)", "2 233", "5.6%"],
      ["4", "Conflict (само)", "2 096", "5.3%"],
      ["5", "Conflict + Dev-Procedural", "1 656", "4.2%"],
      ["6", "Dev-Procedural + Dev-Track/Heading", "1 517", "3.8%"],
      ["7", "Inflight Event/Encounter (само)", "1 488", "3.7%"],
      ["8", "ATC Issue + Conflict + Dev-Proced.", "1 480", "3.7%"],
      ["9", "Dev-Procedural + Ground Incursion", "1 332", "3.3%"],
      ["10", "Deviation - Procedural (само)", "1 220", "3.1%"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Deviation - Procedural участва в 8 от 10-те най-чести комбинации, потвърждавайки нейната доминация. Само две от топ-10 комбинациите не включват Deviation - Procedural: Aircraft Equipment Problem (самостоятелна) и Conflict (самостоятелна). Тези наблюдения имат практическо значение за класификацията \u2014 модел, който подценява честотата на Deviation - Procedural, ще пропуска голям процент от верните етикети.")));

  // 2.2
  p.push(heading2(t("2.2. Стратифицирано семплиране и разделяне на данните")));
  p.push(bodyParagraph([
    run(t("От 172 183 доклада е извлечена стратифицирана извадка от 39 894 доклада чрез ")),
    en(t("MultilabelStratifiedShuffleSplit")),
    run(t(" с ")),
    en(t("random_state=42")),
    run(t(". Извадката е разделена на обучаващ набор от 31 850 доклада и тестов набор от 8 044 доклада. Тестовият набор е замразен и използван за оценка на всички експерименти.")),
  ]));

  const tNum2 = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNum2}. Разделяне на данните`))]));
  p.push(makeDataTable(
    ["Набор", "Брой доклади", "Използване"],
    [
      ["Обучаващ (Train)", "31 850", "Обучение на ML + фино настройване на LLM"],
      ["Тестов (Test)", "8 044", "Оценка на всички модели (замразен)"],
      ["Пълен (Full)", "164 139", "Допълн. експеримент с ML на 164K"],
    ],
  ));
  p.push(emptyLine());
  // Train/test split per-category table
  const tNumTT = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumTT}. Разпределение на категориите в обучаващия и тестовия набор`))]));
  p.push(makeDataTable(
    ["Категория", "Train", "Train%", "Test", "Test%"],
    [
      ["ATC Issue", "5 464", "17.2%", "1 371", "17.0%"],
      ["Aircraft Equipment Problem", "9 157", "28.8%", "2 297", "28.6%"],
      ["Airspace Violation", "1 270", "4.0%", "318", "4.0%"],
      ["Conflict", "8 597", "27.0%", "2 156", "26.8%"],
      ["Deviation - Altitude", "5 268", "16.5%", "1 322", "16.4%"],
      ["Deviation - Procedural", "20 914", "65.7%", "5 246", "65.2%"],
      ["Deviation - Speed", "929", "2.9%", "233", "2.9%"],
      ["Deviation - Track/Heading", "3 764", "11.8%", "944", "11.7%"],
      ["Flight Deck/Cabin Event", "2 282", "7.2%", "573", "7.1%"],
      ["Ground Event/Encounter", "2 644", "8.3%", "663", "8.2%"],
      ["Ground Excursion", "691", "2.2%", "173", "2.2%"],
      ["Ground Incursion", "2 340", "7.3%", "587", "7.3%"],
      ["Inflight Event/Encounter", "7 180", "22.5%", "1 801", "22.4%"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Процентното разпределение на категориите е почти идентично между обучаващия и тестовия набор (максимално отклонение < 0.3%), което потвърждава успешното стратифицирано семплиране. Статистиката на тестовия набор (средно 2.20 етикета на доклад, медиана 2) също е практически идентична с тази на обучаващия набор (средно 2.21, медиана 2). Единствена разлика е, че максималният брой етикети в тестовия набор е 7 (спрямо 9 в обучаващия), което отразява естествената вариация при семплирането на екстремните случаи.")));
  p.push(bodyParagraph([
    run(t("Допълнителен набор от данни е създаден за експеримента с пълния корпус от 164 139 доклада (172 183 минус 8 044 тестови). Този набор включва всички доклади, които не са в тестовия набор, и се използва за проверка дали увеличаването на обучаващите данни подобрява резултатите. Резултатите показват, че по-големият набор увеличава ")),
    en(t("Recall")),
    run(t(" при всички категории, но намалява ")),
    en(t("Precision")),
    run(t(", водейки до по-нисък нетен Macro-F1.")),
  ]));

  // 2.3
  p.push(heading2(t("2.3. Класическо машинно обучение")));
  p.push(heading3(t("2.3.1. TF-IDF векторизация")));
  p.push(bodyParagraph([
    run(t("Текстовите наративи са преобразувани в числови признаци чрез TF-IDF векторизатор с максимум 50 000 признака, обхват на н-грами (1, 2) и ")),
    en(t("sublinear_tf=True")),
    run(t(' (логаритмично скалиране на честотата на термините). Тази конфигурация улавя както отделни думи (униграми), така и двусловни фрази (биграми), което е важно за авиационния домейн \u2014 например \u201Egear up\u201C (вдигнато колесно шаси) носи различно значение от отделните думи.')),
  ]));
  p.push(bodyParagraph([
    run(t("Проведена е аблация на TF-IDF параметрите с 8 конфигурации (вариации на ")),
    en(t("max_features")),
    run(t(", ")),
    en(t("ngram_range")),
    run(t(" и ")),
    en(t("sublinear_tf")),
    run(t(") чрез 3-кратна кръстосана валидация. Всички конфигурации попадат в диапазон от 0.005 Macro-F1 (0.6248–0.6296), което показва, че TF-IDF параметрите имат пренебрежимо влияние.")),
  ]));

  p.push(heading3(t("2.3.2. XGBoost класификатор")));
  p.push(bodyParagraph([
    run(t("За всяка от 13-те категории е обучен независим бинарен XGBoost класификатор (")),
    en(t("Binary Relevance")),
    run(t(" подход). Основните параметри са: 300 дървета, максимална дълбочина 6, скорост на обучение 0.1, ")),
    en(t("tree_method=hist")),
    run(t(" (за ефективност) и автоматично ")),
    en(t("scale_pos_weight")),
    run(t(" за компенсация на класовия дисбаланс.")),
  ]));
  p.push(bodyParagraph([
    run(t("XGBoost използва хистограмно базирано изграждане на дървета (")),
    en(t("tree_method=hist")),
    run(t("), което значително ускорява обучението при 50 000 признака \u2014 от ~400 секунди на класификатор (без hist) на ~250 секунди (с hist). Параметърът ")),
    en(t("scale_pos_weight")),
    run(t(" се изчислява автоматично за всяка от 13-те категории като съотношение между негативните и позитивните примери, компенсирайки класовия дисбаланс.")),
  ]));
  p.push(bodyParagraph(t("Допълнително е проведено сравнение с LinearSVC и Logistic Regression чрез RandomizedSearchCV (30 итерации, 3-кратна кръстосана валидация). XGBoost превъзхожда и двата линейни модела по Macro-F1, потвърждавайки избора на класификатор. LinearSVC постига най-висок Micro-F1 (0.750), но е слаб по Macro-F1 (0.655) \u2014 това показва, че моделът предпочита честите категории за сметка на рядките. Logistic Regression (C=1.45, L2 регуларизация, SAGA оптимизатор) е по-балансиран, но все пак по-слаб от XGBoost.")));

  const tNum3 = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNum3}. Сравнение на класически модели (32K обучаващ набор)`))]));
  p.push(makeDataTable(
    ["Модел", "CV Macro-F1", "Test Macro-F1", "Test Micro-F1"],
    [
      ["XGBoost", "0.679", "0.691", "0.746"],
      ["Logistic Regression", "0.504", "0.670", "0.738"],
      ["LinearSVC", "0.473", "0.655", "0.750"],
    ],
  ));
  p.push(emptyLine());

  // 2.4
  p.push(heading2(t("2.4. Подходи с големи езикови модели")));

  p.push(heading3(t("2.4.1. Избор на модели")));
  p.push(bodyParagraph([
    run(t("Изборът на модели преминава през три итерации. Първоначалният избор е ")),
    en(t("Meta LLaMA 3.1 8B Instruct")),
    run(t(", но достъпът до модела изисква одобрение (gate approval), което забавя работата. Втората итерация е ")),
    en(t("Ministral 3-8B-Instruct")),
    run(t(" от Mistral AI — малък мултимодален модел с 8 милиарда параметъра. Ministral обаче се оказва проблематичен за фино настройване: моделът е съхранен като ")),
    en(t("Mistral3ForConditionalGeneration")),
    run(t(" с FP8 квантизация, което предотвратява прилагането на стандартен QLoRA (4-битова NF4 квантизация). LoRA адаптерите се обучават върху FP8 базов модел, но резултатите не показват подобрение — Macro-F1 дори спада от 0.491 (zero-shot) на 0.489 (fine-tuned).")),
  ]));
  p.push(bodyParagraph([
    run(t("Третата и финална итерация е ")),
    en(t("Qwen/Qwen3-8B")),
    run(t(" от Alibaba — чист текстов авторегресивен модел (")),
    en(t("CausalLM")),
    run(t(") с 8 милиарда параметъра, лиценз Apache 2.0 и без ограничения за достъп. Qwen3-8B поддържа стандартен QLoRA workflow без проблеми и разполага с вграден режим на разсъждение (")),
    en(t("thinking mode")),
    run(t("), активиран чрез параметъра ")),
    en(t("enable_thinking=True")),
    run(t(". Резултатите от Ministral 8B са запазени и анализирани за сравнение, но Qwen3-8B е основният малък модел за всички последващи експерименти.")),
  ]));
  p.push(bodyParagraph(t("Освен малките модели (8B параметъра), експериментите включват два големи MoE модела. Mistral Large 3 е водещият модел на Mistral AI с 675 милиарда общи параметъра и 41 милиарда активни при всяко извикване. Той е достъпен безплатно чрез Batch API на Mistral, което го прави изключително привлекателен за изследователски цели. DeepSeek V3.2 от DeepSeek AI е подобен MoE модел с 671 милиарда параметъра, достъпен чрез DeepInfra API с поддръжка на кеширане на префикси.")));

  const tNum4 = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNum4}. Използвани модели`))]));
  p.push(makeDataTable(
    ["Модел", "Тип", "Параметри", "Лиценз", "Инфраструктура"],
    [
      ["XGBoost", "Градиентни дървета", "N/A", "Apache 2.0", "Локален CPU"],
      ["Qwen3-8B", "CausalLM", "8B", "Apache 2.0", "Modal (L4/A100)"],
      ["Ministral 8B", "Мултимодален", "8B", "Apache 2.0", "Modal (L4/A100)"],
      ["Mistral Large 3", "MoE", "675B (41B акт.)", "Apache 2.0", "Mistral Batch API"],
      ["DeepSeek V3.2", "MoE", "671B", "Proprietary", "DeepInfra API"],
    ],
  ));
  p.push(emptyLine());

  p.push(heading3(t("2.4.2. Дизайн на промптове")));
  p.push(bodyParagraph(t("Използвани са два типа промптове:")));
  p.push(bodyParagraph([
    bold(t("Основен промпт (basic): ")),
    run(t("Кратка системна инструкция с изброяване на 13-те категории и указание да се върне JSON масив.")),
  ]));
  p.push(bodyParagraph([
    bold(t("Обогатен с таксономия промпт (taxonomy): ")),
    run(t("Разширена системна инструкция с подкатегориите на NASA ASRS, описания и дискриминативни подсказки. Например, пояснява разликата между Ground Excursion (напускане на предвидената повърхност) и Ground Incursion (неоторизирано навлизане), която е основен източник на грешки.")),
  ]));
  p.push(bodyParagraph(t('Потребителското съобщение за всички експерименти е: \u201EClassify this ASRS report into applicable anomaly categories:\u201C последвано от наратива на доклада.')));
  p.push(bodyParagraph(t("Дизайнът на промптовете преминава през итеративен процес. Основният промпт е минималистичен \u2014 списък от 13 категории с инструкция за JSON масив. Таксономично обогатеният промпт е разработен въз основа на официалната таксономия на NASA ASRS и включва три ключови елемента: (1) подкатегориите на всяка основна категория; (2) кратки описания, пояснявящи обхвата; (3) дискриминативни подсказки, разграничаващи лесно объркваеми категории.")));
  p.push(bodyParagraph(t("Изборът на формат за изхода (JSON масив) е мотивиран от лесната парсируемост и еднозначността \u2014 масивът от низове не допуска двусмислие. Алтернативни формати (свободен текст, разделени със запетая) биха изисквали по-сложна нормализация и биха увеличили процента на грешки при парсването.")));

  p.push(heading3(t("2.4.3. Нулево обучение (Zero-Shot)")));
  p.push(bodyParagraph(t("При нулевото обучение моделът получава само системната инструкция и текста на доклада, без примери. Моделът трябва да идентифицира приложимите категории единствено въз основа на знанията, придобити при предобучението. Тестван е с основен и обогатен с таксономия промпт.")));

  p.push(heading3(t("2.4.4. Обучение с малко примери (Few-Shot)")));
  p.push(bodyParagraph(t("При few-shot подхода промптът включва по 2–3 примера за всяка категория (общо 26–39 примера), избрани от обучаващия набор. Стратегията за избор предпочита доклади с по-малко етикети (по-ясен сигнал) и по-кратък текст (за спестяване на контекстен бюджет). Наративите на примерите са съкратени до 600 знака, а тестовите доклади — до 1500 знака.")));

  p.push(bodyParagraph(t("Стратегията за съкращаване на наративите е важен компромис. Примерите се съкращават до 600 знака (запазвайки най-информативното начало на наратива), а тестовите доклади \u2014 до 1 500 знака. Тази стратегия позволява включването на 39 примера (3 × 13 категории) в промпта, без да се надхвърли контекстният прозорец. При Mistral Large 3 (128K контекст) съкращаването е по-малко критично, а при Qwen3-8B (16K за few-shot) \u2014 от решаващо значение.")));
  p.push(bodyParagraph(t("Изборът на примери предпочита доклади с по-малко етикети (по-ясен сигнал за конкретната категория) и по-кратък текст (за спестяване на контекстен бюджет). Тази евристика е по-ефективна от случайния избор, тъй като осигурява ясни, недвусмислени примери за всяка категория.")));

  p.push(heading3(t("2.4.5. Фино настройване чрез QLoRA")));
  p.push(bodyParagraph([
    run(t("Фино настройването е приложено върху Qwen3-8B чрез QLoRA с 4-битова NF4 квантизация. Обучението обхваща 2 епохи върху 31 850 обучаващи доклада. Всеки тренировъчен пример е форматиран като чат разговор: системна инструкция → потребителско съобщение с наратива → отговор на асистента с JSON масив от категории. Параметрите на обучението са: ")),
    en(t("batch_size")),
    run(t(" = 4, ")),
    en(t("gradient_accumulation")),
    run(t(" = 4, ")),
    en(t("learning_rate")),
    run(t(" = 2e-5, ")),
    en(t("cosine")),
    run(t(" планировчик с warmup 5%.")),
  ]));

  // QLoRA training config table
  const tNumQLora = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumQLora}. Конфигурация на QLoRA обучението`))]));
  p.push(makeDataTable(
    ["Параметър", "Стойност"],
    [
      ["Базов модел", "Qwen/Qwen3-8B"],
      ["Квантизация", "4-bit NF4 (double quantization)"],
      ["LoRA ранг (r)", "16"],
      ["LoRA alpha", "16"],
      ["Целеви модули", "q_proj, v_proj"],
      ["Dropout", "0.05"],
      ["Епохи", "2"],
      ["Размер на партида", "4"],
      ["Gradient accumulation", "4"],
      ["Ефективна партида", "16"],
      ["Скорост на обучение", "2e-5"],
      ["Планировчик", "cosine (warmup 5%)"],
      ["Оптимизатор", "paged_adamw_8bit"],
      ["Максимална дължина", "1 024 токена"],
      ["Прецизност на изчисленията", "BFloat16"],
      ["GPU", "NVIDIA A100 (80 GB)"],
      ["Общо стъпки", "3 982"],
      ["Крайна загуба", "1.691"],
      ["Точност на токените", "66.8%"],
      ["Време за обучение", "3 ч. 47 мин."],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Обучението преминава през 3 982 стъпки (2 епохи × 31 850 примера / ефективна партида 16). Загубата намалява стабилно от ~2.5 в началото до 1.691 в края, а точността на токените (процент на правилно предсказани следващи токени) достига 66.8%. Тези стойности показват успешна конвергенция, но не перфектна — моделът все още прави грешки при генериране на отговорите.")));
  p.push(bodyParagraph([
    run(t("Всеки тренировъчен пример е форматиран като чат разговор с три съобщения: (1) системна инструкция (същата като при zero-shot), (2) потребителско съобщение с наратива на доклада (съкратен до 1 500 знака) и (3) отговор на асистента с коректния JSON масив от категории. Форматирането използва ")),
    en(t("apply_chat_template()")),
    run(t(" с ")),
    en(t("enable_thinking=False")),
    run(t(", за да се избегне генерирането на thinking блокове по време на обучението.")),
  ]));

  p.push(heading3(t("2.4.6. Режим на разсъждение (Thinking Mode)")));
  p.push(bodyParagraph([
    run(t("Режимът на разсъждение е тестван с два модела: Qwen3-8B (")),
    en(t("enable_thinking=True")),
    run(t(") и DeepSeek V3.2 (")),
    en(t("reasoning=True")),
    run(t("). В този режим моделът генерира междинни стъпки на разсъждение преди крайния отговор, което увеличава значително броя на генерираните токени (10–27 пъти повече). При Qwen3-8B разсъжденията се появяват в ")),
    en(t("<think>")),
    run(t(" блокове, които се премахват преди парсването. При DeepSeek V3.2 разсъжденията се връщат в отделно поле ")),
    en(t("reasoning_content")),
    run(t(".")),
  ]));

  p.push(heading3(t("2.4.7. Парсване на изхода от LLM")));
  p.push(bodyParagraph(t("Всички LLM експерименти използват тристепенна стратегия за парсване на генерирания изход. Първо се опитва директен JSON парсинг на целия изход. Ако това не успее, се търси JSON масив чрез регулярен израз (първият [...] блок в текста). Ако и това не успее, се прилага размито (fuzzy) търсене — за всяка категория се проверява дали нейното име (без разлика в регистъра) се среща като подниз в изхода.")));
  p.push(bodyParagraph(t("При моделите с режим на разсъждение (Qwen3-8B thinking) генерираният текст съдържа <think>...</think> блокове с вътрешни разсъждения. Тези блокове се премахват чрез регулярен израз преди парсването. При DeepSeek V3.2 разсъжденията се връщат в отделно поле (reasoning_content), а полето content съдържа чист JSON, който не се нуждае от допълнителна обработка.")));
  p.push(bodyParagraph(t("Mistral Large 3 обгражда JSON изхода в markdown код блокове (```json ... ```), които се премахват преди парсването. Тези особености на различните модели изискват адаптивен парсинг, който е имплементиран в общата функция parse_llm_output().")));
  p.push(bodyParagraph(t("Нормализацията на категориите (функция _normalize()) преобразува парсираните низове към точните имена на категориите чрез съвпадение без разлика в регистъра. За Mistral Large 3, който понякога връща формат с подкатегория (напр. Aircraft Equipment Problem: Less Severe), нормализацията премахва суфикса след двоеточието.")));

  p.push(heading3(t("2.4.8. Инфраструктура за LLM инференция")));
  p.push(bodyParagraph([
    run(t("За самохоствани модели (Qwen3-8B, Ministral 8B) е използвана платформата Modal за облачно GPU изчисление. Инференцията е проведена чрез ")),
    en(t("vLLM")),
    run(t(" — високопроизводителна библиотека за LLM сървинг с поддръжка на PagedAttention, continuous batching и LoRA адаптери. Ключовите параметри на vLLM включват gpu_memory_utilization (дял от GPU паметта за KV кеша), max_model_len (максимална дължина на последователността) и batch_size (брой заявки в една партида).")),
  ]));
  p.push(bodyParagraph(t("За API-базираните модели (Mistral Large 3, DeepSeek V3.2) е използван различен подход. Mistral Large 3 е достъпен чрез Batch API на Mistral, който обработва хиляди заявки паралелно без ограничение на честотата (rate limiting). DeepSeek V3.2 е достъпен чрез DeepInfra API с OpenAI-съвместим интерфейс и поддръжка на 50 паралелни заявки чрез aiohttp. И двата API използват кеширане на префикси (prefix caching) — идентичните системни промптове при поредни заявки водят до 62–82% кеш попадения, което значително намалява разходите.")));

  // 2.5
  p.push(heading2(t("2.5. Разширение към подкатегории (48 етикета)")));
  p.push(bodyParagraph(t("Освен 13-те основни категории, е изследвано и класифициране на 48 подкатегории — например Aircraft Equipment Problem се разделя на Critical и Less Severe, а Ground Event/Encounter — на 8 подкатегории. За тази задача е използван отделен набор данни от 40 106 доклада (32 089 обучаващи и 8 017 тестови), отново със стратифицирано семплиране.")));
  p.push(bodyParagraph(t("Преминаването от 13 към 48 подкатегории увеличава значително сложността на задачата. Броят на уникалните комбинации от етикети нараства многократно, а някои подкатегории са изключително редки. Например, подкатегорията Weather/Turbulence за Ground Event има само 25 тестови примера, а Ground Equipment Issue — само 45. При толкова малък брой примери дори перфектно обучен класификатор може да покаже нестабилни F1 резултати.")));
  p.push(bodyParagraph(t("Разпределението на подкатегориите е още по-небалансирано от това на основните категории. Най-честата подкатегория (Deviation - Procedural: Published Material/Policy) се среща в над 30% от докладите, докато най-рядката (Weather/Turbulence за Ground Event) — в по-малко от 0.1%. Това съотношение на дисбаланс от над 300:1 е значително предизвикателство за всички класификатори.")));
  p.push(bodyParagraph(t("За LLM моделите подкатегорийната задача е допълнително усложнена от необходимостта да разграничават между 48 различни етикета в един промпт. Контекстният бюджет се натоварва значително — таксономичният промпт за 48 подкатегории е приблизително два пъти по-дълъг от този за 13 категории. При по-малките модели (Qwen3-8B) това води до драматичен спад в представянето.")));
  p.push(bodyParagraph(t("Най-добрите подкатегории по F1 (за Classic ML) са Hazardous Material Violation (F1 = 0.824), Smoke/Fire/Fumes/Odor (F1 = 0.815) и Wake Vortex Encounter (F1 = 0.813) — всички с ясно дефинирани лексикални сигнатури. Най-слабите са Weather/Turbulence за Ground Event (F1 = 0.000), Ground Equipment Issue (F1 = 0.118) и Vehicle (F1 = 0.164) — всички с малко тренировъчни примери и размити лексикални граници.")));

  // 2.6
  p.push(heading2(t("2.6. Метрики за оценка")));
  p.push(bodyParagraph(t("Всички модели са оценени по следните метрики: Precision (точност), Recall (пълнота), F1-Score и ROC-AUC, изчислени както в Macro-усредняване (равна тежест на всички категории), така и в Micro-усредняване (тежест пропорционална на честотата). Основната метрика за сравнение е Macro-F1, тъй като третира еднакво важни и редките категории.")));
  p.push(bodyParagraph([
    en(t("Precision")),
    run(t(" (точност) измерва дела на правилните положителни предсказания от всички положителни предсказания на модела. Висока Precision означава малко фалшиви положителни — моделът рядко грешно приписва категория. ")),
    en(t("Recall")),
    run(t(" (пълнота) измерва дела на намерените положителни примери от всички реални положителни. Висок Recall означава, че моделът рядко пропуска реални категории. ")),
    en(t("F1-Score")),
    run(t(" е хармоничната средна на Precision и Recall, балансираща и двата аспекта.")),
  ]));
  p.push(bodyParagraph([
    run(t("В многоетикетен контекст ")),
    en(t("Macro-F1")),
    run(t(" изчислява F1 за всяка от 13-те категории поотделно и взема аритметичната средна. Категория с 5 246 тестови примера (Deviation - Procedural) и категория с 173 примера (Ground Excursion) имат еднаква тежест. Това е важно за авиационната безопасност, тъй като редките категории (като Ground Excursion) могат да бъдат критично важни. ")),
    en(t("Micro-F1")),
    run(t(" агрегира всички предсказания глобално и дава по-голяма тежест на честите категории — по-добре отразява цялостната производителност.")),
  ]));
  p.push(bodyParagraph([
    en(t("ROC-AUC")),
    run(t(" е особено важна метрика за оценка на класификатори, тъй като не зависи от избрания праг на решение. За Classic ML, който генерира непрекъснати вероятности за всяка категория, AUC измерва способността за ранжиране — дали истинските положителни получават по-високи вероятности от истинските отрицателни. За LLM моделите, които генерират бинарни решения (включена/изключена категория), AUC се изчислява от бинарния вектор на предсказания, което дава по-консервативни стойности.")),
  ]));
  p.push(bodyParagraph(t("Тази асиметрия в AUC изчислението обяснява защо Classic ML постига значително по-висок Macro-AUC (0.932) в сравнение с LLM моделите (0.70–0.81). Classic ML произвежда градирани вероятности, които позволяват по-фино ранжиране, докато LLM произвежда само бинарни включения/изключения, което ограничава AUC стойностите.")));

  // 2.7
  p.push(heading2(t("2.7. Инфраструктура и изчислителни ресурси")));
  p.push(bodyParagraph([
    run(t("Локалната среда е VS Code с Jupyter на машина с Windows 11 и 16 GB RAM (без GPU). Всички LLM експерименти са проведени на платформата Modal — облачен сървис за GPU изчисления. Използвани са GPU модели NVIDIA L4 (24 GB) за инференция и A100 (80 GB) за обучение. Допълнително са използвани API-тата на Mistral (")),
    en(t("Batch API")),
    run(t(") и DeepInfra за моделите Mistral Large 3 и DeepSeek V3.2.")),
  ]));
  p.push(bodyParagraph([
    run(t("Modal е платформа за безсървърно GPU изчисление, която позволява дефиниране на функции с GPU изисквания в Python код. Всяка функция се изпълнява в изолиран контейнер с предварително инсталирани зависимости (")),
    en(t("vLLM")),
    run(t(", ")),
    en(t("transformers")),
    run(t(", ")),
    en(t("bitsandbytes")),
    run(t(", ")),
    en(t("trl")),
    run(t("). Предимствата на Modal включват: автоматично мащабиране на ресурсите, заплащане само за използваното време и бърз старт (< 60 секунди за студен контейнер с кеширани модели). Недостатък е, че при дълготрайни задачи (> 2 часа) връзката с локалния клиент може да прекъсне, изисквайки използването на откачен режим и Modal Volumes за устойчиво съхранение на резултатите.")),
  ]));
  p.push(bodyParagraph([
    run(t("За инференция на самохоствани модели е използвана библиотеката ")),
    en(t("vLLM")),
    run(t(" — високопроизводителен LLM сървър с поддръжка на ")),
    en(t("PagedAttention")),
    run(t(" (ефективно управление на KV кеша), ")),
    en(t("continuous batching")),
    run(t(" (групиране на заявки за максимална GPU утилизация) и динамично зареждане на LoRA адаптери. Ключовите параметри на vLLM конфигурацията включват: ")),
    en(t("gpu_memory_utilization=0.85")),
    run(t(" (85% от GPU паметта за KV кеша), ")),
    en(t("max_model_len=8192")),
    run(t(" (максимална дължина на последователността за стандартен режим) и ")),
    en(t("max_model_len=32768")),
    run(t(" (за режим на разсъждение, който генерира по-дълги изходи). При инференция с LoRA адаптер се добавят ")),
    en(t("enable_lora=True")),
    run(t(" и ")),
    en(t("max_lora_rank=16")),
    run(t(".")),
  ]));

  const tNum5a = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNum5a}. Конфигурация на vLLM инференцията`))]));
  p.push(makeDataTable(
    ["Параметър", "Стандартен режим", "Thinking режим"],
    [
      ["GPU", "NVIDIA L4 (24 GB)", "NVIDIA A100 (80 GB)"],
      ["max_model_len", "8 192", "32 768"],
      ["max_tokens", "256", "4 096"],
      ["gpu_memory_utilization", "0.85", "0.90"],
      ["batch_size", "64", "32"],
      ["temperature", "0", "0"],
      ["enable_thinking", "False", "True"],
      ["tensor_parallel_size", "1", "1"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph([
    run(t("За API-базираните модели са използвани два различни подхода. Mistral Large 3 е достъпен чрез ")),
    en(t("Batch API")),
    run(t(" на Mistral, който приема файл с хиляди заявки и ги обработва асинхронно, без ограничение на честотата (")),
    en(t("rate limiting")),
    run(t("). Този подход позволява обработка на 8 044 доклада за 4–5 минути при нулева цена (безплатен план). Batch API обаче може да остане в състояние ")),
    en(t("QUEUED")),
    run(t(' без да стартира обработка, което изисква преминаване към real-time API като резервен вариант.')),
  ]));
  p.push(bodyParagraph([
    run(t("DeepSeek V3.2 е достъпен чрез DeepInfra API с OpenAI-съвместим интерфейс. Заявките са изпращани асинхронно чрез ")),
    en(t("aiohttp")),
    run(t(" с до 50 паралелни конекции. DeepInfra поддържа кеширане на префикси (")),
    en(t("prefix caching")),
    run(t(") — идентичните системни промптове при поредни заявки водят до 62–82% кеш попадения, което намалява цената на входните токени наполовина. Без кеширане цената на DeepSeek V3.2 би била приблизително двойна.")),
  ]));
  p.push(bodyParagraph(t("За Classic ML експериментите локалната машина с 16 GB RAM е достатъчна за малки модели, но при 50 000 TF-IDF признака XGBoost изисква значителна оперативна памет (~1.5 GB на класификатор). При ниска свободна памет обучението може да генерира грешки при алокация. Поради тази причина хиперпараметричната настройка (с множество едновременни класификатори) е проведена на Modal с 32-ядрен CPU инстанс, осигуряващ 64 GB RAM.")));

  const tNum5 = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNum5}. Обобщение на изчислителните ресурси`))]));
  p.push(makeDataTable(
    ["Ресурс", "Тип", "Приблизителна цена"],
    [
      ["Modal (GPU облак)", "L4, A100, CPU", "~$38"],
      ["DeepInfra API", "DeepSeek V3.2", "~$15"],
      ["Mistral Batch API", "Mistral Large 3", "$0 (безплатен план)"],
      ["Локален CPU", "XGBoost", "$0"],
      ["ОБЩО", "", "~$53"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Общата цена на всички експерименти (~$53) демонстрира, че мащабно сравнително изследване с множество модели е финансово достъпно в академичен контекст. Най-скъпите компоненти са QLoRA обучението (~$10.56 за 3.8 часа на A100) и DeepSeek V3.2 с режим на разсъждение (~$6.73 за 291 минути). Безплатният Batch API на Mistral е ключов за достъпността — без него разходите за Mistral Large 3 биха били значителни ($0.2–$0.5 на 1000 заявки при real-time API).")));

  return p;
}

// ─── CHAPTER 3: RESULTS ─────────────────────────────────────────────────────

function buildChapter3() {
  const p = [];
  p.push(heading1(t("ГЛАВА 3. РЕЗУЛТАТИ")));

  // 3.1 Classic ML
  p.push(heading2(t("3.1. Класическо машинно обучение")));
  p.push(bodyParagraph(t("Класическият подход с TF-IDF и XGBoost постига Macro-F1 = 0.691 и Micro-F1 = 0.746 на тестовия набор от 8 044 доклада. Macro-AUC е 0.932, което показва отлична способност за ранжиране на категориите.")));

  const tNumCML = nextTable();
  const cmlData = loadMetricsCsv("classic_ml_text_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumCML}. Резултати на Classic ML (TF-IDF + XGBoost, 32K)`))]));
  if (cmlData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], cmlData.rows));
  }
  p.push(emptyLine());

  p.push(bodyParagraph(t("Най-силните категории са Aircraft Equipment Problem (F1 = 0.816), Conflict (F1 = 0.801) и Deviation - Procedural (F1 = 0.795). Най-слабите категории са рядките — Airspace Violation (F1 = 0.568), Ground Excursion (F1 = 0.572) и Deviation - Speed (F1 = 0.577), което е типично за класификатори при дисбалансирани данни.")));

  const fCML = nextFigure();
  p.push(imageParagraph("classic_ml_f1_barchart.png", 5.5, 3.5));
  p.push(caption(t(`Фигура ${fCML}. F1 резултати по категории — Classic ML`)));

  p.push(bodyParagraph(t("Обучение върху пълния набор от 164 139 доклада дава малко по-ниски F1 резултати (Macro-F1 = 0.678, Micro-F1 = 0.739), но по-висок Macro-AUC = 0.942. По-големият набор увеличава пълнотата (recall), но намалява точността (precision), тъй като засилва ефекта от класовия дисбаланс.")));

  // Classic ML full 164K per-category
  const tNumCMLFull = nextTable();
  const cmlFullData = loadMetricsCsv("classic_ml_full_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumCMLFull}. Резултати на Classic ML (164K обучаващ набор)`))]));
  if (cmlFullData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], cmlFullData.rows));
  }
  p.push(emptyLine());
  p.push(bodyParagraph(t("При 164K обучаващ набор Recall нараства при всички 13 категории (средно +0.09 спрямо 32K), но Precision спада значително (средно −0.06). Най-драматичният спад на Precision е при Deviation - Speed (от 0.551 на 0.375) и Airspace Violation (от 0.480 на 0.380). Тези категории са рядки и при по-голям обучаващ набор XGBoost \u201Eнаучава\u201C да ги предсказва по-агресивно, увеличавайки фалшивите положителни.")));
  p.push(bodyParagraph(t("Интересно е, че Macro-AUC се увеличава (от 0.932 на 0.942) въпреки по-ниския Macro-F1. Това означава, че по-големият набор подобрява ранжиращата способност на модела, но при фиксиран праг от 0.5 баланс между Precision и Recall е по-неблагоприятен. Оптимизация на праговете за всяка категория може да възстанови или подобри F1 резултатите.")));

  p.push(bodyParagraph(t("Хиперпараметричната настройка потвърждава, че базовите параметри са близки до оптималните — разликата между настроените и базовите резултати е Macro-F1 +0.002 / Micro-F1 −0.001.")));
  p.push(bodyParagraph(t("Забележителна е ниската AUC стойност на Deviation - Procedural (0.794) в сравнение с останалите категории (средно 0.94). Това се дължи на нейната висока честота (65.4%) — при толкова преобладаваща категория дори добрата точност и пълнота дават по-ниска AUC, защото базовата честота (base rate) на положителните примери е висока.")));
  p.push(bodyParagraph(t("Аблацията на TF-IDF параметрите тества 8 конфигурации: вариации на max_features (10K, 25K, 50K, 100K), ngram_range ((1,1) vs (1,2) vs (1,3)) и sublinear_tf (True/False). Всички конфигурации попадат в диапазон от 0.005 Macro-F1 при 3-кратна кръстосана валидация (0.6248–0.6296), което показва забележителна робустност на TF-IDF подхода — крайният резултат е малко чувствителен към точните параметри.")));

  const tNumAbl = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumAbl}. Аблация на TF-IDF параметри (3-кратна кръстосана валидация)`))]));
  p.push(makeDataTable(
    ["Конфигурация", "max_features", "ngram_range", "sublinear_tf", "CV Macro-F1"],
    [
      ["baseline", "50K", "(1,2)", "True", "0.6274"],
      ["no_sublinear", "50K", "(1,2)", "False", "0.6296"],
      ["unigram", "50K", "(1,1)", "True", "0.6248"],
      ["trigram", "50K", "(1,3)", "True", "0.6269"],
      ["10k", "10K", "(1,2)", "True", "0.6279"],
      ["25k", "25K", "(1,2)", "True", "0.6282"],
      ["100k", "100K", "(1,2)", "True", "0.6267"],
      ["uni_nosub", "50K", "(1,1)", "False", "0.6272"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Резултатите показват, че базовата конфигурация с 50K признака и субкосинусова TF е практически оптимална — разликата с най-добрата конфигурация (no_sublinear: 0.6296) е само 0.0022 Macro-F1. Изненадващо, конфигурацията без субкосинусова TF (sublinear_tf=False) е леко по-добра, но разликата е статистически незначима. Биграмите добавят малко, но стабилно подобрение спрямо униграмите (+0.0026), а триграмите не дават допълнителна полза. Намаляването на признаците до 10K не влошава значително резултатите, което показва, че повечето класификационна информация се съдържа в топ 10 000 признака.")));

  // 3.2 Zero-Shot
  p.push(heading2(t("3.2. Резултати от нулево обучение (Zero-Shot)")));

  const tNumZS = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumZS}. Сравнение на zero-shot подходите`))]));
  p.push(makeDataTable(
    ["Модел", "Промпт", "Macro-F1", "Micro-F1", "Macro-AUC"],
    [
      ["Mistral Large 3 (675B)", "taxonomy", "0.658", "0.712", "0.793"],
      ["DeepSeek V3.2 (671B)", "taxonomy", "0.623", "0.693", "0.746"],
      ["DeepSeek V3.2 + thinking", "taxonomy", "0.681", "0.723", "0.810"],
      ["Qwen3-8B", "taxonomy", "0.499", "0.605", "0.701"],
      ["Qwen3-8B", "basic", "0.459", "0.473", "0.727"],
      ["Ministral 8B", "basic", "0.491", "0.543", "0.744"],
    ],
  ));
  p.push(emptyLine());

  p.push(bodyParagraph(t("Обогатяването на промпта с таксономия дава значително подобрение за малките модели — Qwen3-8B показва Macro-F1 +0.040 и Micro-F1 +0.133 с таксономичния промпт спрямо основния. За големите модели (Mistral Large 3, DeepSeek V3.2) таксономичният промпт е стандартен за всички експерименти.")));
  p.push(bodyParagraph(t("Режимът на разсъждение при DeepSeek V3.2 (671B) подобрява Macro-F1 с +0.058 (от 0.623 на 0.681), което е най-добрият LLM резултат по тази метрика. Подобрението обаче е свързано с 45 пъти по-дълго време на работа и 4.8 пъти по-висока цена.")));
  p.push(bodyParagraph(t("Анализът по категории разкрива, че режимът на разсъждение при DeepSeek V3.2 е особено ефективен за категории, изискващи контекстуален анализ. ATC Issue показва подобрение от +0.160 F1 (от 0.376 на 0.536), а Airspace Violation — +0.126 F1 (от 0.460 на 0.586). Тези категории изискват разбиране на регулации и процедури, което се улеснява от стъпково разсъждение.")));
  p.push(bodyParagraph(t("Mistral Large 3 постига Macro-F1 = 0.658 при zero-shot чрез безплатния Batch API, което го прави изключително практичен избор. Моделът показва най-висока F1 по Deviation - Procedural (0.793), вероятно защото неговите обширни обучаващи данни включват познание за авиационните процедури. Gratis достъпът чрез Batch API (обработка на 8 044 доклада за 5 минути) прави този подход идеален за бързо прототипиране.")));

  // Mistral Large ZS per-category
  const tNumMistralZS = nextTable();
  const mistralZSData = loadMetricsCsv("mistral_large_zs_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumMistralZS}. Резултати по категории — Mistral Large 3 Zero-Shot`))]));
  if (mistralZSData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], mistralZSData.rows));
  }
  p.push(emptyLine());

  // DeepSeek V3.2 + thinking per-category
  const tNumDSThink = nextTable();
  const dsThinkData = loadMetricsCsv("deepseek_v32_thinking_parent_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumDSThink}. Резултати по категории — DeepSeek V3.2 Zero-Shot + Thinking`))]));
  if (dsThinkData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], dsThinkData.rows));
  }
  p.push(emptyLine());

  p.push(bodyParagraph(t("DeepSeek V3.2 с режим на разсъждение показва балансиран профил — Precision и Recall са близки за повечето категории, което е признак за добра калибрация. Conflict постига най-висок F1 (0.837), следван от Aircraft Equipment Problem (0.820). Най-слабата категория е ATC Issue (0.536) с ниска Precision (0.441), което показва, че моделът тенденциозно предсказва тази категория.")));

  // Qwen3-8B ZS taxonomy per-category
  const tNumQwenZST = nextTable();
  const qwenZSTData = loadMetricsCsv("zero_shot_taxonomy_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumQwenZST}. Резултати по категории \u2014 Qwen3-8B Zero-Shot Taxonomy`))]));
  if (qwenZSTData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], qwenZSTData.rows));
  }
  p.push(emptyLine());
  p.push(bodyParagraph(t("Qwen3-8B с таксономичен промпт показва профил с висока Precision за някои категории (Aircraft Equipment Problem: 0.876, Conflict: 0.677) и висок Recall за други (Deviation - Procedural: 0.876, Ground Event: 0.647). Най-драматичното подобрение спрямо основния промпт е при Deviation - Procedural, която скача от F1 = 0.353 (basic) на F1 = 0.770 (taxonomy) — подобрение от +0.417 F1 пункта.")));
  p.push(bodyParagraph(t("Подобрението при Deviation - Procedural се дължи на бележката в таксономичния промпт: \"This is the broadest category (~65% of reports)\" и инструкцията \"when in doubt, include it\". Без тази бележка моделът подценява честотата на категорията и предсказва само 23.3% Recall (спрямо 87.6% с таксономия). Този пример илюстрира силата на промпт инженерството — малко добавена информация може драматично да промени поведението на модела.")));

  // DeepSeek V3.2 non-thinking per-category
  const tNumDSNT = nextTable();
  const dsNTData = loadMetricsCsv("deepseek_v32_parent_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumDSNT}. Резултати по категории \u2014 DeepSeek V3.2 Zero-Shot (без разсъждение)`))]));
  if (dsNTData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], dsNTData.rows));
  }
  p.push(emptyLine());
  p.push(bodyParagraph(t("Сравнението между DeepSeek V3.2 без и с режим на разсъждение разкрива интересни разлики. Без разсъждение моделът показва висока Precision за Aircraft Equipment Problem (0.874) и Conflict (0.876), но нисък Recall за ATC Issue (0.308) и Airspace Violation (0.352). Режимът на разсъждение увеличава Recall значително за тези категории (ATC Issue: от 0.308 на 0.685, Airspace Violation: от 0.352 на 0.572), което показва, че допълнителното размишление помага на модела да идентифицира неочевидните случаи.")));

  // Qwen3-8B basic ZS per-category
  const tNumQwenBasic = nextTable();
  const qwenBasicData = loadMetricsCsv("zero_shot_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumQwenBasic}. Резултати по категории \u2014 Qwen3-8B Zero-Shot (basic промпт)`))]));
  if (qwenBasicData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], qwenBasicData.rows));
  }
  p.push(emptyLine());
  p.push(bodyParagraph(t("Qwen3-8B с основен промпт показва изключително небалансиран профил. Flight Deck/Cabin Event има Precision само 0.135, но Recall 0.803 \u2014 моделът масово предсказва тази категория. Deviation - Procedural има висока Precision (0.727), но нисък Recall (0.233) \u2014 моделът подценява честотата на категорията. Тези дисбаланси са пряко следствие от липсата на таксономична информация в основния промпт.")));
  p.push(bodyParagraph(t("Сравнението между basic и taxonomy промпт за Qwen3-8B разкрива, че таксономичният промпт подобрява 10 от 13 категории по F1, като най-голямото подобрение е при Deviation - Procedural (+0.417 F1) и Flight Deck/Cabin Event (+0.310 F1). Трите категории, които не се подобряват, са Aircraft Equipment Problem, Conflict и Deviation - Speed \u2014 категории с достатъчно ясни лексикални сигнатури, за които допълнителната таксономична информация не е необходима.")));

  // 3.3 Few-Shot
  p.push(heading2(t("3.3. Резултати от обучение с малко примери (Few-Shot)")));

  const tNumFS = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumFS}. Сравнение на few-shot подходите`))]));
  p.push(makeDataTable(
    ["Модел", "Промпт", "Macro-F1", "Micro-F1", "Macro-AUC"],
    [
      ["Mistral Large 3 (675B)", "taxonomy", "0.640", "0.686", "0.793"],
      ["Qwen3-8B + thinking", "taxonomy", "0.533", "0.556", "0.705"],
      ["Qwen3-8B", "taxonomy", "0.526", "0.544", "0.706"],
      ["Ministral 8B", "basic", "0.540", "0.536", "0.746"],
      ["Qwen3-8B", "basic", "0.453", "0.468", "0.704"],
    ],
  ));
  p.push(emptyLine());

  p.push(bodyParagraph(t("Интересен е фактът, че few-shot подходът не винаги превъзхожда zero-shot. Mistral Large 3 показва по-ниски резултати при few-shot (Macro-F1 0.640) спрямо zero-shot (0.658). Това може да се дължи на факта, че примерите заемат контекстен бюджет и потенциално насочват модела към определени шаблони вместо да се възползва от цялата си предобучена преценка.")));
  p.push(bodyParagraph(t("Режимът на разсъждение при Qwen3-8B дава маргинално подобрение: Macro-F1 +0.007, Micro-F1 +0.013 спрямо few-shot без разсъждение. Това подобрение е на цена от ~$6.67 (спрямо ~$0.45 без разсъждение), което прави подхода неефективен за 8B модели.")));
  p.push(bodyParagraph(t("При Qwen3-8B few-shot с thinking mode, 99.6% от изходите съдържат <think> блокове със средна дължина 2 986 знака и максимална — 15 945 знака. Парсването е успешно за всичките 8 025 доклада — 7 990 чрез директен JSON, 4 чрез regex, 31 чрез fuzzy matching и 0 празни. Времето за обработка обаче е 144 минути на A100 (спрямо ~34 минути за few-shot без thinking на L4), тъй като режимът на разсъждение генерира 10–27 пъти повече токени.")));

  // Mistral Large few-shot per-category
  const tNumMistralFS = nextTable();
  const mistralFSData = loadMetricsCsv("mistral_large_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumMistralFS}. Резултати по категории — Mistral Large 3 Few-Shot`))]));
  if (mistralFSData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], mistralFSData.rows));
  }
  p.push(emptyLine());
  p.push(bodyParagraph(t("Сравнението между zero-shot и few-shot за Mistral Large 3 показва, че примерите увеличават точността (Precision) за повечето категории, но намаляват пълнотата (Recall). Например Aircraft Equipment Problem преминава от Precision 0.819/Recall 0.812 (ZS) на 0.918/0.640 (FS) — примерите правят модела по-предпазлив, но и по-склонен да пропуска. Нетният ефект е отрицателен за Macro-F1 (−0.018).")));

  // Qwen3-8B FS taxonomy + thinking per-category
  const tNumQwenFSTh = nextTable();
  const qwenFSThData = loadMetricsCsv("few_shot_taxonomy_thinking_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumQwenFSTh}. Резултати по категории \u2014 Qwen3-8B Few-Shot Taxonomy + Thinking`))]));
  if (qwenFSThData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], qwenFSThData.rows));
  }
  p.push(emptyLine());
  p.push(bodyParagraph(t("Анализът по категории разкрива, че thinking mode при Qwen3-8B подобрява определени категории, но влошава други. Най-голямо подобрение е при Aircraft Equipment Problem (+0.249 F1 спрямо few-shot без thinking: от 0.424 на 0.673), което показва, че разсъждението помага за идентифицирането на технически проблеми. Най-голям спад е при Deviation - Track/Heading (−0.177 F1: от 0.495 на 0.318), където допълнителните разсъждения могат да объркат модела.")));
  p.push(bodyParagraph(t("Средната дължина на thinking блоковете е 2 986 знака (медиана 2 513, максимум 15 945). Тази значителна дължина показва, че моделът наистина извършва многостъпково разсъждение, но качеството на това разсъждение е ограничено от 8B размера \u2014 моделът генерира мисли, но те не винаги са коректни или полезни за класификацията.")));

  // 3.4 Fine-Tuned
  p.push(heading2(t("3.4. Резултати от фино настройване")));
  p.push(bodyParagraph(t("Фино настройването на Qwen3-8B чрез QLoRA дава Macro-F1 = 0.510 и Micro-F1 = 0.632. Сравнено с нулевото обучение (Micro-F1 = 0.473 за basic, 0.605 за taxonomy), фино настройването подобрява значително Micro-F1, но Macro-F1 остава по-нисък от zero-shot с таксономичен промпт.")));

  const tNumFT = nextTable();
  const ftData = loadMetricsCsv("finetune_metrics.csv");
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumFT}. Резултати от фино настройване — Qwen3-8B QLoRA`))]));
  if (ftData.rows.length > 0) {
    p.push(makeDataTable(["Категория", "Precision", "Recall", "F1", "ROC-AUC"], ftData.rows));
  }
  p.push(emptyLine());

  p.push(bodyParagraph(t("Най-силната категория е Aircraft Equipment Problem (F1 = 0.783), а най-слабата — Airspace Violation (F1 = 0.120), което показва, че фино настройването не решава проблема с редките категории. Обучението отнема 3 часа и 47 минути на A100 GPU и струва ~$10.56.")));

  p.push(bodyParagraph(t('Опитът с Ministral 8B показва, че фино настройването на модел с FP8 квантизация (вместо истинско QLoRA) не дава подобрение \u2014 Macro-F1 намалява от 0.491 (zero-shot) на 0.489 (fine-tuned). Моделът се превръща в \u201Eмашина за утвърждаване\u201C с висока пълнота, но много ниска точност.')));

  // Ministral comparison table
  const tNumMin = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumMin}. Резултати на Ministral 8B (архивирани)`))]));
  p.push(makeDataTable(
    ["Подход", "Macro-F1", "Micro-F1", "Macro-AUC"],
    [
      ["Zero-Shot (basic)", "0.491", "0.543", "0.744"],
      ["Few-Shot (basic)", "0.540", "0.536", "0.746"],
      ["Fine-Tuned (LoRA/FP8)", "0.489", "0.542", "0.744"],
    ],
  ));
  p.push(emptyLine());
  p.push(bodyParagraph(t("Сравнението между двата 8B модела показва, че Ministral 8B постига по-добри zero-shot резултати (Macro-F1 0.491 vs 0.459 за Qwen3-8B с basic промпт), но по-ниски резултати при few-shot (0.540 vs 0.526 за Qwen3-8B с таксономичен промпт). Ключовата разлика е при фино настройването — Qwen3-8B с истинско QLoRA (4-bit NF4) постига значително подобрение (Macro-F1 0.510, Micro-F1 0.632), докато Ministral 8B с LoRA върху FP8 не показва подобрение (Macro-F1 0.489). Това потвърждава важността на правилната квантизационна стратегия за фино настройване.")));
  p.push(bodyParagraph(t("Интересна находка е, че фино настройването на Qwen3-8B подобрява значително Micro-F1 (+0.159 спрямо basic zero-shot), но Macro-F1 остава по-нисък от zero-shot с таксономичен промпт (0.510 vs 0.499). Причината е, че фино настройването подобрява предимно честите категории (Aircraft Equipment Problem, Deviation - Procedural, Conflict), но влошава рядките (Airspace Violation спада до F1 = 0.120 от 0.335 при ZS taxonomy). Micro-F1, претеглен по честота, се подобрява, но Macro-F1, третиращ всички категории еднакво, остава нисък поради провала при рядките категории.")));
  p.push(bodyParagraph(t("Общата цена на фино настройването е $10.83 — $10.56 за обучение на A100 GPU и $0.27 за инференция на L4 GPU. За сравнение, zero-shot с таксономичен промпт струва $0.33 и дава сравним Macro-F1. Това поставя под въпрос рентабилността на фино настройването за малки модели при наличие на добре конструирани промптове.")));
  p.push(bodyParagraph(t("Анализът на кривата на обучение показва, че загубата намалява стабилно през двете епохи — от ~2.5 в началото до ~1.95 в края на първата епоха и до 1.691 в края на втората. Точността на токените следва обратна крива — от ~58% в началото до ~64% след първата епоха и 66.8% накрая. Липсата на рязко влошаване (отскачане на загубата) показва, че моделът не е преобучен, но бавната конвергенция подсказва, че допълнителни епохи биха могли да подобрят резултатите.")));
  p.push(bodyParagraph(t("Профилът на грешките на фино настроения модел се различава качествено от този на zero-shot модела. При zero-shot (basic) Qwen3-8B имa тенденция да надпредсказва Flight Deck/Cabin Event (Precision 0.135, Recall 0.803) и да подпредсказва Deviation - Procedural (Precision 0.727, Recall 0.233). След фино настройване моделът показва по-балансиран профил за честите категории (Aircraft Equipment Problem: P=0.789, R=0.778; Conflict: P=0.654, R=0.690), но остава изключително слаб за рядките (Airspace Violation: P=0.098, R=0.157; Ground Excursion: P=0.185, R=0.197).")));

  // 3.5 Grand Comparison
  p.push(heading2(t("3.5. Общо сравнение на всички подходи")));

  const tNumGrand = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumGrand}. Обобщено сравнение на всички експерименти (13 категории)`))]));
  p.push(makeDataTable(
    ["Модел", "Подход", "Macro-F1", "Micro-F1", "Macro-AUC"],
    [
      ["Classic ML (32K)", "TF-IDF+XGB", "0.691", "0.746", "0.932"],
      ["DeepSeek V3.2 + thinking", "ZS taxonomy", "0.681", "0.723", "0.810"],
      ["Classic ML (164K)", "TF-IDF+XGB", "0.678", "0.739", "0.942"],
      ["Mistral Large 3", "ZS taxonomy", "0.658", "0.712", "0.793"],
      ["Mistral Large 3", "FS taxonomy", "0.640", "0.686", "0.793"],
      ["DeepSeek V3.2", "ZS taxonomy", "0.623", "0.693", "0.746"],
      ["Ministral 8B", "FS basic", "0.540", "0.536", "0.746"],
      ["Qwen3-8B + thinking", "FS taxonomy", "0.533", "0.556", "0.705"],
      ["Qwen3-8B", "FS taxonomy", "0.526", "0.544", "0.706"],
      ["Qwen3-8B", "Fine-tuned", "0.510", "0.632", "0.700"],
      ["Qwen3-8B", "ZS taxonomy", "0.499", "0.605", "0.701"],
      ["Ministral 8B", "ZS basic", "0.491", "0.543", "0.744"],
      ["Qwen3-8B", "ZS basic", "0.459", "0.473", "0.727"],
      ["Qwen3-8B", "FS basic", "0.453", "0.468", "0.704"],
    ],
  ));
  p.push(emptyLine());

  const fGrand = nextFigure();
  p.push(imageParagraph("fig_grand_comparison.png", 5.5, 4.0));
  p.push(caption(t(`Фигура ${fGrand}. Сравнение Macro-F1 / Micro-F1 на всички модели`)));

  const fApproach = nextFigure();
  p.push(imageParagraph("fig_approach_summary.png", 5.5, 3.5));
  p.push(caption(t(`Фигура ${fApproach}. Средни резултати по подход`)));

  const fHeatmap = nextFigure();
  p.push(imageParagraph("fig_category_heatmap.png", 5.5, 4.5));
  p.push(caption(t(`Фигура ${fHeatmap}. Топлинна карта на F1 по категории и модели`)));

  p.push(bodyParagraph(t("Фигура " + fGrand + " визуализира всички модели по Macro-F1 и Micro-F1. Ясно се очертават три групи: (1) Classic ML с най-високи резултати; (2) големи LLM модели (Mistral Large, DeepSeek + thinking) в средата; (3) малки LLM модели (Qwen3-8B, Ministral 8B) с най-ниски резултати.")));
  p.push(bodyParagraph(t("Фигура " + fApproach + " агрегира резултатите по подход (zero-shot, few-shot, fine-tuned, classic ML) и показва, че средният zero-shot Macro-F1 (0.547) е по-висок от средния few-shot (0.538), което е контраинтуитивно. Обяснението е, че few-shot примерите могат да бъдат контрапродуктивни — те заемат контекстен бюджет, ограничават дължината на тестовия наратив и понякога насочват модела към специфични шаблони вместо към обща генерализация.")));
  p.push(bodyParagraph(t("Топлинната карта (Фигура " + fHeatmap + ") разкрива интересни модели. Aircraft Equipment Problem и Conflict са последователно силни категории за всички подходи — техните лексикални сигнатури (технически проблеми, конфликти в трафика) са ясно разпознаваеми. Deviation - Procedural е силна за Classic ML и някои LLM модели, но слаба за малки модели без таксономичен промпт — широката дефиниция на категорията затруднява моделите без допълнителна информация.")));
  p.push(bodyParagraph(t("Най-информативната находка от общото сравнение е, че разликата между най-добрия LLM подход (DeepSeek V3.2 + thinking: Macro-F1 0.681) и Classic ML (0.691) е само 0.010 Macro-F1 пункта. Тази малка разлика обаче маскира съществени различия в профила на грешките: Classic ML е по-балансиран (по-високи резултати за рядките категории), докато LLM моделите са по-силни за категории, изискващи семантично разбиране (ATC Issue, Airspace Violation с thinking mode).")));

  // 3.6 Scale Effect
  p.push(heading2(t("3.6. Ефект от мащаба на модела")));

  const fScale = nextFigure();
  p.push(imageParagraph("fig_scale_effect.png", 5.5, 3.5));
  p.push(caption(t(`Фигура ${fScale}. Ефект от мащаба: Dense vs MoE архитектури`)));

  p.push(bodyParagraph(t("Фигура " + fScale + " показва зависимостта между размера на модела и Macro-F1 при zero-shot класификация с таксономичен промпт. Малките модели (8B) постигат Macro-F1 около 0.49–0.50, докато големите MoE модели (671–675B) достигат 0.62–0.68. Увеличаването на мащаба от 8B на 675B подобрява Macro-F1 с ~0.16 абсолютни пункта.")));
  p.push(bodyParagraph(t("Важно уточнение е, че големите модели (Mistral Large 3 и DeepSeek V3.2) са с архитектура Mixture of Experts (MoE), при която само част от параметрите се активира при всяко извикване. Mistral Large 3 активира 41B от 675B параметъра, а DeepSeek V3.2 — подобна пропорция от 671B. Въпреки това, дори ефективният размер на активните параметри (41B) е значително по-голям от 8B, което обяснява разликата в качеството.")));
  p.push(bodyParagraph(t("Скалиращите закони (scaling laws) на LLM предсказват, че качеството на модела расте логаритмично с броя на параметрите и обема на обучаващите данни. Резултатите от настоящата работа са съвместими с тази теория — увеличаването на параметрите от 8B на 41B активни (5×) дава подобрение от ~0.16 Macro-F1, а не петкратно подобрение. Тази субкосинусова зависимост означава, че всяко следващо удвояване на параметрите дава все по-малък прираст в качеството.")));
  p.push(bodyParagraph(t("Ефектът от мащаба не е линеен — удвояването на параметрите не удвоява качеството. Скокът от 8B на 41B активни параметъра (5x) дава подобрение от ~0.16 Macro-F1 (от ~0.50 на ~0.66). Допълнителното увеличение от мащаба (пълните 671-675B параметъра) допринася предимно чрез по-богата вътрешна репрезентация и по-добро покритие на предобучаващия корпус.")));

  // 3.7 Subcategory
  p.push(heading2(t("3.7. Резултати по подкатегории (48 етикета)")));

  const tNumSub = nextTable();
  p.push(emptyLine());
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumSub}. Сравнение на модели по 48 подкатегории`))]));
  p.push(makeDataTable(
    ["Модел", "Macro-F1", "Micro-F1", "Macro-AUC"],
    [
      ["Classic ML (XGBoost)", "0.510", "0.600", "0.934"],
      ["Mistral Large 3 ZS", "0.449", "0.494", "0.744"],
      ["DeepSeek V3.2 ZS", "0.422", "0.456", "0.708"],
      ["DeepSeek V3.2 ZS + thinking", "0.419", "0.466", "0.690"],
      ["Qwen3-8B ZS", "0.235", "0.304", "0.629"],
    ],
  ));
  p.push(emptyLine());

  p.push(bodyParagraph(t("Преминаването от 13 към 48 подкатегории намалява Macro-F1 с −0.181 за Classic ML (от 0.691 на 0.510). AUC остава почти непроменен (0.934 vs 0.932), което показва запазена способност за ранжиране. За LLM моделите спадът е още по-драстичен: Qwen3-8B пада от 0.499 на 0.235.")));

  const fSubGrand = nextFigure();
  p.push(imageParagraph("fig_sub_grand_comparison.png", 5.5, 3.5));
  p.push(caption(t(`Фигура ${fSubGrand}. Сравнение по 48 подкатегории`)));

  const fParentSub = nextFigure();
  p.push(imageParagraph("fig_parent_vs_sub.png", 5.5, 3.5));
  p.push(caption(t(`Фигура ${fParentSub}. Сравнение: 13 категории vs 48 подкатегории`)));

  p.push(bodyParagraph(t("Режимът на разсъждение при DeepSeek V3.2 не подобрява, а дори влошава резултатите при 48-те подкатегории (Macro-F1 0.419 vs 0.422 без разсъждение). Причината е, че разсъжденческите токени изчерпват лимита от 4096 токена, оставяйки без място за JSON отговора — 21.6% от отговорите са празни.")));
  p.push(bodyParagraph(t("Mistral Large 3 показва най-добрите LLM резултати при подкатегориите (Macro-F1 0.449). Моделът е особено силен при ясно дефинирани подкатегории като Passenger Misconduct (F1 = 0.876), Smoke/Fire/Fumes/Odor (F1 = 0.825) и Hazardous Material Violation (F1 = 0.791). Интересно е, че Mistral Large 3 превъзхожда Classic ML при 11 от 48 подкатегории, включително Landing Without Clearance (+0.194 F1), Gear Up Landing (+0.196 F1) и UAS (+0.177 F1) — всички категории, при които семантичното разбиране е по-важно от статистическите шаблони.")));
  p.push(bodyParagraph(t("Qwen3-8B показва драматичен спад при подкатегориите — от Macro-F1 0.499 (13 категории) на 0.235 (48 подкатегории), спад от 0.264 абсолютни пункта. Най-силните подкатегории за модела са Smoke/Fire/Fumes/Odor (F1 = 0.673) и Passenger Misconduct (F1 = 0.596), които имат ясни лексикални маркери. Най-слабите са CFTT/CFIT (F1 = 0.005), Undershoot (F1 = 0.015) и Ground Equipment Issue (F1 = 0.029).")));
  p.push(bodyParagraph(t("Ключовият извод от подкатегорийния анализ е, че сложността на класификационната задача нараства нелинейно с броя на етикетите. Преминаването от 13 на 48 етикета не просто увеличава шанса за грешка — то фундаментално променя характера на задачата. При 48 подкатегории повечето етикети са редки, границите между тях са размити, а контекстният прозорец на промпта се натоварва значително. Classic ML запазва сравнително висок Macro-AUC (0.934), показвайки, че ранжирането на подкатегориите остава надеждно, дори когато бинарните решения (включена/изключена) са по-трудни.")));

  return p;
}

// ─── CHAPTER 4: DISCUSSION ──────────────────────────────────────────────────

function buildChapter4() {
  const p = [];
  p.push(heading1(t("ГЛАВА 4. ДИСКУСИЯ")));

  // 4.1
  p.push(heading2(t("4.1. Защо класическото машинно обучение превъзхожда LLM")));
  p.push(bodyParagraph(t("Класическият подход с TF-IDF + XGBoost (Macro-F1 = 0.691) постига значително по-високи резултати от всички LLM подходи (най-добър: 0.681 с DeepSeek V3.2 + thinking). Това противоречи на популярните очаквания, че LLM моделите ще превъзхождат традиционните методи във всички текстови задачи. Няколко фактора обясняват тази разлика.")));
  p.push(bodyParagraph(t("Първо, TF-IDF извлича домейн-специфични н-грами, които са силно информативни за класификацията. Авиационните съкращения (acft, rwy, apch, twr) и техническите фрази (gear up, runway incursion, flight level) са ясно свързани с определени категории. XGBoost ефективно улавя тези връзки и изгражда нелинейни модели на решение.")));
  p.push(bodyParagraph(t("Второ, Binary Relevance подходът позволява на всеки класификатор да се настрои индивидуално за своята категория, включително чрез scale_pos_weight за компенсация на дисбаланса. LLM моделите решават всички 13 категории едновременно в един промпт, без възможност за такава индивидуална настройка.")));
  p.push(bodyParagraph(t("Трето, XGBoost е обучен върху 31 850 примера от целевата задача, докато LLM моделите (освен при fine-tuning) разчитат на знания от предобучението. Дори при fine-tuning, QLoRA адаптира по-малко от 1% от параметрите на модела, което ограничава степента на специализация.")));
  p.push(bodyParagraph(t("Четвърто, LLM моделите генерират изход в текстов формат (JSON масив), който трябва да бъде парсиран. Макар и процентът на грешки при парсването да е нисък (0–0.6%), самият процес на генериране е стохастичен — моделът решава последователно кои категории да включи, което може да доведе до нестабилност при гранични случаи. XGBoost, от друга страна, дава детерминистични вероятности за всяка категория.")));
  p.push(bodyParagraph(t("Пето, Macro-AUC на Classic ML (0.932) е значително по-висок от всички LLM модели (0.70–0.81). Това означава, че ранжирането на категориите от XGBoost е много по-надеждно. Ако се приложи оптимален праг за всяка категория (вместо фиксиран праг 0.5), Classic ML може да покаже още по-добри F1 резултати.")));
  p.push(bodyParagraph(t("Шесто, XGBoost е по-устойчив към дисбаланса на класовете благодарение на параметъра scale_pos_weight, който автоматично увеличава тежестта на рядките положителни примери. При Ground Excursion (2.2% от данните) scale_pos_weight е приблизително 45, компенсирайки рядкостта на категорията. LLM моделите нямат подобен механизъм — при zero-shot те разчитат на вътрешната си калибрация, която може да не отразява специфичното разпределение на ASRS данните.")));
  p.push(bodyParagraph(t("Седмо, TF-IDF представянето е подходящо за авиационния домейн поради неговия специализиран лексикон. Думи като runway incursion, gear up, altitude deviation са директно свързани с конкретни категории. Биграмите (н-грами от 2 думи) са особено информативни — например gear up е силен индикатор за Aircraft Equipment Problem, докато runway incursion — за Ground Incursion. Общоцелевите LLM модели може да не придават същата тежест на тези термини.")));
  p.push(bodyParagraph(t("Важно е да се отбележи, че превъзходството на Classic ML не е универсално — то е специфично за ASRS задачата с 31 850 тренировъчни примера и ясно дефинирани категории. В ситуации с по-малко размечени данни (стотици вместо хиляди) или при нужда от бърза адаптация към нови категории, LLM подходите биха имали значително предимство.")));

  // 4.2
  p.push(heading2(t("4.2. Ефективност на промпт инженерството")));
  p.push(bodyParagraph(t("Обогатяването на промпта с таксономична информация е един от най-ефективните подходи с най-ниска цена. За Qwen3-8B промяната от основен към таксономичен промпт дава подобрение от +0.040 Macro-F1 и +0.133 Micro-F1 при zero-shot, без допълнителни разходи за обучение или по-мощно хардуерно оборудване.")));
  p.push(bodyParagraph(t("Таксономичният промпт е особено ефективен за категории, които са лесно объркваеми — например Deviation - Procedural скача от F1 = 0.353 (basic) на F1 = 0.770 (taxonomy), защото дискриминативните подсказки в промпта пояснават нейната широка дефиниция. Подобно, Ground Event преминава от F1 = 0.312 на F1 = 0.405.")));
  p.push(bodyParagraph(t("Тази находка има практически импликации: преди да се инвестира във фино настройване ($10+ за 8B модел), е по-ефективно да се оптимизира промптът, който може да даде сравними или дори по-добри резултати при нулева допълнителна цена.")));
  p.push(bodyParagraph(t("Конкретните елементи на таксономичния промпт, които допринасят за подобрението, включват: (1) изброяването на подкатегориите помага на модела да разбере обхвата на всяка основна категория; (2) дискриминативните подсказки (Ground Excursion = aircraft LEAVING vs Ground Incursion = unauthorized ENTRY) директно адресират най-честите объркванията; (3) бележката за Deviation - Procedural (the broadest category, ~65% of reports) калибрира очакванията на модела за честотата на категорията.")));
  p.push(bodyParagraph(t("Интересно е, че промпт инженерството има различен ефект при различните мащаби на модела. За малките модели (8B) таксономичният промпт дава значително подобрение, тъй като компенсира ограниченото знание от предобучението. За големите модели (671-675B), които вече имат обширно знание за авиационния домейн, таксономичният промпт служи повече за структуриране на отговора, отколкото за добавяне на нова информация.")));
  p.push(bodyParagraph(t("Few-shot примерите, за разлика от промпт обогатяването, дават смесени резултати. За Ministral 8B few-shot подобрява Macro-F1 с +0.049 (от 0.491 на 0.540), но за Mistral Large 3 — влошава с −0.018 (от 0.658 на 0.640). Хипотезата е, че примерите фиксират модела към определени шаблони, което помага на по-слабите модели, но ограничава по-мощните. Освен това, примерите заемат контекстен бюджет, принуждавайки тестовите наративи да бъдат съкратени до 1 500 знака.")));

  // 4.3
  p.push(heading2(t("4.3. Мащаб на модела срещу техника")));
  p.push(bodyParagraph(t("Резултатите разкриват интересна динамика между размера на модела и използваната техника. По-големите модели (671–675B) постигат значително по-добри резултати от малките (8B) при zero-shot — разликата е ~0.16 Macro-F1 пункта. Същевременно, фино настройването на малък модел (8B) дава Macro-F1 = 0.510, което е по-ниско от zero-shot на голям модел (Mistral Large 3: 0.658).")));
  p.push(bodyParagraph(t("Това води до важен извод: при ограничен бюджет е по-добре да се използва по-голям модел чрез API (Mistral Large 3 е безплатен чрез Batch API), отколкото да се фино настройва малък модел. Единственият случай, когато fine-tuning на малък модел е обоснован, е при нужда от офлайн работа без достъп до API.")));
  p.push(bodyParagraph(t("Режимът на разсъждение добавя допълнително измерение на анализа. При 671B модел (DeepSeek V3.2) thinking mode дава +0.058 Macro-F1, което е най-голямото подобрение от единична техника в изследването. Но при 8B модел (Qwen3-8B) thinking mode дава само +0.007 Macro-F1 — маргинално подобрение на цена от $6.67 (спрямо $0.45 без thinking). Тази асиметрия показва, че разсъждението е ефективно само когато моделът има достатъчно знание за обработка — малките модели генерират мисли, но тези мисли не са достатъчно информативни.")));
  p.push(bodyParagraph(t("Практическата йерархия на подходите, подредена по Macro-F1 / цена, е: (1) Classic ML: $0, Macro-F1 0.691; (2) Mistral Large 3 ZS чрез безплатен Batch API: $0, Macro-F1 0.658; (3) DeepSeek V3.2 ZS: $1.39, Macro-F1 0.623; (4) Qwen3-8B ZS taxonomy: $0.33, Macro-F1 0.499. Фино настройването ($10.83 за Macro-F1 0.510) и thinking mode ($6.67 за Macro-F1 0.533) имат слаба рентабилност в сравнение с тези алтернативи.")));

  // 4.4 Cost-Performance
  p.push(heading2(t("4.4. Анализ на съотношението цена — ефективност")));

  const fCost = nextFigure();
  p.push(imageParagraph("fig_cost_vs_performance.png", 5.5, 4.0));
  p.push(caption(t(`Фигура ${fCost}. Цена vs ефективност на всички подходи`)));

  p.push(bodyParagraph(t("Фигура " + fCost + " визуализира съотношението между цена и Macro-F1 за всички експерименти. Classic ML доминира с $0 цена и най-висок Macro-F1. Mistral Large 3 чрез безплатния Batch API предлага отличен баланс — Macro-F1 = 0.658 без разходи. DeepSeek V3.2 с режим на разсъждение постига най-добрия LLM резултат (0.681), но на цена от $6.73.")));
  p.push(bodyParagraph(t("Фино настройването е най-скъпият LLM подход ($10.56 за обучение + $0.27 за инференция), но дава средни резултати (Macro-F1 = 0.510). Режимът на разсъждение при малък модел (Qwen3-8B) е неефективен — $6.67 за маргинално подобрение от +0.007 Macro-F1.")));
  p.push(bodyParagraph(t("Общата цена на всички експерименти е около $53 — $38 за Modal (GPU облак), $15 за DeepInfra (DeepSeek V3.2 API) и $0 за Mistral Large 3 (безплатен Batch API). Това демонстрира, че мащабно сравнително изследване е финансово достъпно благодарение на облачните GPU услуги и безплатните API нива. Класическият ML подход (XGBoost) може да работи дори на обикновен лаптоп без GPU.")));
  p.push(bodyParagraph(t("Времевата ефективност също е в полза на Classic ML — обучението отнема около 55 минути на CPU, а инференцията на 8 044 доклада — секунди. За сравнение, QLoRA обучението отнема 3 часа и 47 минути на A100, а DeepSeek V3.2 с режим на разсъждение — 291 минути за инференция на същия тестов набор.")));
  p.push(bodyParagraph(t("Разходите за инференция варират значително между подходите. Classic ML инференцията е практически безплатна — веднъж обучен, XGBoost класифицира 8 044 доклада за по-малко от 5 секунди на обикновен CPU. vLLM инференцията на L4 GPU за Qwen3-8B без thinking струва ~$0.33–$0.46 за 8K доклада (24–34 минути). API инференцията за DeepSeek V3.2 без thinking струва $1.39 за 6.5 минути, а с thinking — $6.73 за 291 минути. Тези разлики от порядъци подчертават предимството на Classic ML за производствени приложения.")));
  p.push(bodyParagraph(t("Интересен е случаят с Mistral Large 3, който предлага безплатна инференция чрез Batch API. Този модел постига Macro-F1 = 0.658 при нулева цена, което го прави идеален за сценарии, при които Classic ML не е приложим (например при липса на размечени данни за обучение). Ограничението е, че безплатният план има месечен лимит от 4 милиона токена, което е достатъчно за ~8 000 доклада, но не за по-голям мащаб на производствено приложение.")));
  p.push(bodyParagraph(t("Анализът на маргиналната стойност на допълнителните разходи е показателен. QLoRA обучението струва $10.56 и дава Macro-F1 = 0.510. Същият или по-добър Macro-F1 (0.499) може да бъде постигнат чрез таксономичен промпт при zero-shot за $0.33 — разликата от 32 пъти в цената при сравним резултат поставя под въпрос стойността на фино настройването за малки модели.")));

  // 4.5
  p.push(heading2(t("4.5. Трудност на категориите")));
  p.push(bodyParagraph(t("Определени категории последователно се оказват трудни за всички модели. Airspace Violation е трудна за LLM моделите (F1 = 0.120–0.516), вероятно поради нейната специфичност — описанията на нарушения на въздушното пространство изискват точно познание на регулациите. Ground Excursion и Deviation - Speed също показват ниски резултати поради рядкостта им (2.2% и 2.9% от данните).")));
  p.push(bodyParagraph(t("От друга страна, Aircraft Equipment Problem и Conflict последователно показват високи F1 резултати (0.72–0.84) при всички подходи, вероятно защото описанията на технически проблеми и конфликти в трафика съдържат ясни ключови думи и фрази.")));
  p.push(bodyParagraph(t("Deviation - Procedural е уникална категория — най-честа (65.4%) и с най-широка дефиниция. Classic ML постига висок F1 (0.795) поради множеството тренировъчни примери, но LLM моделите се справят различно: Mistral Large 3 постига 0.793 (сравним с ML), докато Qwen3-8B с basic промпт — само 0.353 (значително по-ниско). Таксономичният промпт драматично подобрява тази категория за малките модели (от 0.353 на 0.770 за Qwen3-8B ZS), тъй като бележката „This is the broadest category (~65% of reports)\" помага на модела да калибрира прага.")));
  p.push(bodyParagraph(t("Flight Deck/Cabin Event е интересен случай — LLM моделите (особено Mistral Large 3 с F1 = 0.660–0.693) имат по-ниски резултати от Classic ML (0.738), но фино настроеният Qwen3-8B показва особено нисък резултат (0.359). Това може да се дължи на факта, че тази категория включва разнородни събития (заболяване, поведение на пътници, дим/огън), чиито лексикални сигнатури са различни.")));
  p.push(bodyParagraph(t("При подкатегориите трудността се увеличава значително. Категории с много малко примери (Weather/Turbulence за Ground Event — само 25 тестови примера) показват F1 = 0.000 дори за Classic ML. Същевременно, ясно дефинирани подкатегории с достатъчно примери (Hazardous Material Violation, Smoke/Fire/Fumes/Odor) постигат F1 > 0.80. Това подчертава критичната роля на честотата на категорията за качеството на класификацията.")));
  p.push(bodyParagraph(t("Трудността на категориите се определя от комбинацията от три фактора: честота (рядките категории имат по-малко тренировъчни примери), лексикална специфичност (категории с ясни ключови думи са по-лесни) и семантична сложност (категории с широки или припокриващи се дефиниции са по-трудни). Aircraft Equipment Problem е лесна по и трите фактора — честа, с ясни технически термини, с конкретна дефиниция. Airspace Violation е трудна поради рядкост и нужда от регулаторно познание. Deviation - Procedural е парадоксално трудна въпреки честотата — нейната изключително широка дефиниция я прави трудна за разграничаване от другите категории.")));

  // 4.6
  p.push(heading2(t("4.6. Ограничения и заплахи за валидността")));
  p.push(bodyParagraph(t("Настоящото изследване има няколко ограничения. Първо, използван е само текстовият наратив, без структурираните полета (тип въздухоплавателно средство, фаза на полета, височина), които биха могли да подобрят класификацията. Второ, fine-tuning е тестван само на Qwen3-8B — по-голям модел може да покаже различни резултати. Трето, оценката е върху един замразен тестов набор — кръстосана валидация би дала по-стабилни оценки.")));
  p.push(bodyParagraph(t("Друго ограничение е, че LLM моделите получават наратива като чист текст, без форматиране или структура, които биха могли да помогнат за идентифицирането на ключови елементи. Също така, температурата на генериране е фиксирана на 0 за всички експерименти, а различни стойности биха могли да дадат различни резултати.")));
  p.push(bodyParagraph(t("Заплаха за външната валидност е, че резултатите са специфични за ASRS данни и 13-те категории — генерализацията към други системи за класификация на инциденти не е тествана.")));
  p.push(bodyParagraph(t("Относно LLM моделите, тестваните версии са от определен момент (февруари 2026 г.) и бъдещи версии на същите модели могат да покажат различни резултати. Също така, затворените модели (DeepSeek V3.2 чрез DeepInfra) не предоставят пълна прозрачност за своите обучаващи данни, което поражда въпрос дали ASRS данните не са част от предобучаващия корпус.")));
  p.push(bodyParagraph(t("Откъм методология, Binary Relevance подходът за Classic ML не моделира зависимостите между категориите. Например, ако доклад описва конфликт, вероятността за Deviation - Procedural е по-висока. Моделиране на тези зависимости (чрез Classifier Chains или невронни мрежи с общ репрезентационен слой) може да подобри резултатите.")));
  p.push(bodyParagraph(t("Накрая, оценката е базирана на единичен фиксиран тестов набор от 8 044 доклада. Кръстосана валидация (k-fold cross-validation) би дала по-стабилни оценки и би позволила статистическо тестване на значимостта на разликите между моделите.")));
  p.push(bodyParagraph(t("Специфично ограничение на LLM подхода е стохастичността на генерирането, дори при temperature=0. Различни версии на vLLM, различни GPU и различни размери на батча могат да дадат леко различни резултати поради числените апроксимации при FP16/BF16 аритметика. Тези различия обикновено са незначителни (< 0.5% F1), но означават, че точната възпроизводимост на резултатите изисква фиксиране на всички софтуерни версии и хардуерна конфигурация.")));
  p.push(bodyParagraph(t("Ограничение на Binary Relevance подхода за Classic ML е предположението за независимост между категориите. В реалността категориите на ASRS са корелирани — ако доклад описва Conflict, вероятността за Deviation - Procedural е по-висока (72% от Conflict докладите имат и Deviation - Procedural). Моделиране на тези зависимости чрез Classifier Chains, невронни мрежи с общ репрезентационен слой или графови модели може потенциално да подобри резултатите.")));
  p.push(bodyParagraph(t("Важно ограничение на настоящата работа е липсата на анализ на обяснимостта. За производствено приложение в авиационната безопасност не е достатъчно моделът да даде коректна класификация — необходимо е и обяснение защо доклад е класифициран по определен начин. Classic ML позволява частична обяснимост чрез TF-IDF тежестите на най-информативните думи, а LLM в режим на разсъждение предоставя текстово обяснение, но качеството и надеждността на тези обяснения не са оценени в настоящата работа.")));

  return p;
}

// ─── CONCLUSION ─────────────────────────────────────────────────────────────

function buildConclusion() {
  const p = [];
  p.push(heading1(t("ЗАКЛЮЧЕНИЕ")));

  p.push(bodyParagraph(t("Настоящата магистърска теза представя систематично сравнение на четири подхода за многоетикетна класификация на 172 183 доклада от системата за авиационна безопасност ASRS на NASA по 13 категории аномалии. Основните изводи са следните.")));

  p.push(bodyParagraph([
    bold(t("1. Класическото машинно обучение остава златен стандарт. ")),
    run(t("TF-IDF + XGBoost постига Macro-F1 = 0.691 и Micro-F1 = 0.746, превъзхождайки всички LLM подходи. Доминацията се дължи на домейн-специфичните н-грами, индивидуалната настройка за всяка категория и обучението върху 31 850 размечени примера.")),
  ]));

  p.push(bodyParagraph([
    bold(t("2. Промпт инженерството е най-ефективният подход с ниска цена. ")),
    run(t("Обогатяването на промпта с таксономия подобрява Macro-F1 с +0.040 и Micro-F1 с +0.133 при Qwen3-8B, без допълнителни разходи. Тази техника трябва да бъде първата стъпка преди фино настройване.")),
  ]));

  p.push(bodyParagraph([
    bold(t("3. Мащабът на модела има значение. ")),
    run(t("Големите MoE модели (671–675B) постигат Macro-F1 = 0.62–0.68 при zero-shot, докато малките (8B) — 0.49–0.50. При ограничен бюджет е по-ефективно да се използва голям модел чрез API, отколкото да се фино настройва малък.")),
  ]));

  p.push(bodyParagraph([
    bold(t("4. Фино настройването помага, но не е достатъчно. ")),
    run(t("QLoRA фино настройването подобрява Micro-F1 с +0.159 спрямо zero-shot, но Macro-F1 (0.510) остава далеч под Classic ML (0.691). Режимът на разсъждение дава маргинално подобрение (+0.007 F1) на 8B модели при висока цена ($6.67).")),
  ]));

  p.push(bodyParagraph([
    bold(t("5. Разширението към подкатегории е значително по-трудно. ")),
    run(t("При 48 подкатегории Macro-F1 на Classic ML спада от 0.691 на 0.510, а на LLM моделите — още повече. AUC обаче остава висок (0.934), което означава запазена способност за ранжиране.")),
  ]));

  p.push(emptyLine());
  p.push(bodyParagraph([
    bold(t("Практическа препоръка: ")),
    run(t("За автоматична класификация на ASRS доклади се препоръчва класически ML подход с TF-IDF + XGBoost. Ако се предпочита LLM подход, трябва да се използва голям модел (Mistral Large 3 или по-нов) с таксономично обогатен промпт, като инженерството на промптове е от решаващо значение.")),
  ]));

  p.push(bodyParagraph(t("Общият резултат от изследването разкрива парадокс на съвременното NLP: въпреки впечатляващия напредък на LLM моделите в общоцелеви задачи за обработка на естествен език, специализираните класически подходи запазват своето предимство в структурирани класификационни задачи с ясно дефинирани категории и достатъчно размечени данни. Големите езикови модели обаче предлагат уникално предимство — способност за работа без размечени данни (zero-shot) или с минимален брой примери (few-shot), което ги прави ценни в ситуации, когато размечените данни са ограничени или скъпи за получаване.")));
  p.push(bodyParagraph(t("Йерархията на подходите по ефективност е: Classic ML (TF-IDF + XGBoost) > LLM Zero-Shot на голям модел (671-675B) > LLM Fine-tuned на малък модел (8B) > LLM Zero/Few-Shot на малък модел (8B). Тази йерархия се запазва и при 48 подкатегории, макар и с по-ниски абсолютни стойности.")));
  p.push(bodyParagraph(t("От практическа гледна точка, изследването демонстрира, че цялостно сравнително проучване с множество модели и подходи е финансово достъпно (обща цена ~$53), благодарение на облачните GPU услуги (Modal), безплатните API нива (Mistral) и конкурентните API цени (DeepInfra). Това отваря възможности за подобни изследвания в академичен контекст, без нужда от собствена GPU инфраструктура.")));
  p.push(bodyParagraph(t("Изследването също така демонстрира значимостта на методологическата строгост при сравнението на модели. Използването на един и същ замразен тестов набор за всички 22 експеримента, единна методология за метрики и систематичен контрол на променливите (едни и същи промптове, едни и същи примери) осигуряват надеждност на сравненията. Без тази строгост, разликите между моделите могат да се дължат на случайни фактори, а не на реални различия в качеството.")));
  p.push(bodyParagraph(t("Резултатите имат импликации за индустрията на авиационната безопасност. За автоматична класификация на доклади в производствена среда, Classic ML подходът е оптимален — бърз, евтин, точен и лесен за обясняване. LLM моделите обаче могат да играят допълнителна роля като втори мнение или като инструмент за тriage на новопостъпилите доклади, особено в случаи, когато категориите се променят или се добавят нови.")));

  p.push(heading2(t("Насоки за бъдещо развитие")));
  p.push(bodyParagraph(t("Бъдещите изследвания могат да разширят настоящата работа в няколко направления:")));
  p.push(bodyParagraph(t("Първо, комбиниране на текстови и структурирани признаци (тип въздухоплавателно средство, фаза на полет, метеорологични условия) за мултимодална класификация. ASRS докладите съдържат богата структурирана информация, която не е използвана в настоящата работа. Включването на тези признаци може да подобри резултатите, особено за категории, зависещи от контекста (фаза на полета за Deviation - Altitude, тип въздухоплавателно средство за Aircraft Equipment Problem).")));
  p.push(bodyParagraph(t("Второ, тестване на по-нови и по-мощни модели (GPT-4o, Claude, Gemini) за потенциално по-добри zero-shot резултати. Тези модели имат по-голям контекстен прозорец и вероятно по-добро покритие на авиационния домейн в предобучаващите си данни. Особено интересно би било сравнението с Claude, който показва силни резултати при задачи за класификация и следване на инструкции.")));
  p.push(bodyParagraph(t("Трето, приложение на ансамблови методи, комбиниращи предсказанията на Classic ML и LLM модели. Тъй като двата типа модели имат различни профили на грешки (Classic ML е по-добър за рядки категории, LLM — за семантично сложни), ансамбълът може да превъзхожда всеки отделен модел. Прости стратегии като мажоритарно гласуване или претеглено осредняване на вероятностите са лесно приложими.")));
  p.push(bodyParagraph(t("Четвърто, изследване на активно обучение (active learning), при което LLM предлага етикети за нови доклади, а експерт ги потвърждава или коригира. Този подход може значително да намали натоварването на експертите при класификация на нови доклади, като LLM служи като филтър на първо ниво.")));
  p.push(bodyParagraph(t("Пето, анализ на обяснимостта (explainability) на класификациите — кои части от наратива са най-информативни за всяка категория. За Classic ML това може да се постигне чрез SHAP стойности за TF-IDF признаците. За LLM моделите режимът на разсъждение вече предоставя частична обяснимост чрез генерираните вътрешни мисли.")));
  p.push(bodyParagraph(t("Шесто, тестване на BERT-базирани модели (BERT, RoBERTa, DeBERTa) като допълнителен подход между Classic ML и LLM. Тези модели могат да бъдат фино настроени ефективно на единична GPU и предоставят контекстуално осведомени текстови представяния, които улавят семантичните връзки, недостъпни за TF-IDF.")));
  p.push(bodyParagraph(t("Седмо, кръстосана валидация (k-fold cross-validation) вместо единичен train/test split, за по-стабилни оценки и статистическо тестване на значимостта на разликите между моделите.")));
  p.push(bodyParagraph(t("Осмо, изследване на влиянието на дължината на наратива върху качеството на класификацията. Настоящата работа не анализира дали по-дългите доклади (> 500 думи) се класифицират по-добре от кратките (< 100 думи). Такъв анализ би помогнал за разбирането на минималното количество текст, необходимо за надеждна автоматична класификация.")));
  p.push(bodyParagraph(t("В по-широк план, настоящата работа демонстрира, че систематичното сравнение на различни подходи е от изключителна важност преди вземането на решение за производствено внедряване. Интуитивното очакване, че по-новите и по-мощни LLM модели ще превъзхождат традиционните методи, не се потвърждава в конкретния контекст на ASRS класификацията. Този резултат подчертава необходимостта от емпирична валидация вместо разчитане на общи бенчмаркове, които може да не отразяват специфичните характеристики на целевата задача и домейн.")));

  return p;
}

// ─── BIBLIOGRAPHY ───────────────────────────────────────────────────────────

function buildBibliography() {
  const p = [];
  p.push(heading1(t("БИБЛИОГРАФИЯ")));

  const refs = [
    "[1] Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523.",
    "[2] Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
    "[3] Tsoumakas, G. & Katakis, I. (2007). Multi-label classification: An overview. International Journal of Data Warehousing and Mining, 3(3), 1–13.",
    "[4] Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.",
    "[5] Brown, T. et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems, 33, 1877–1901.",
    "[6] Wei, J. et al. (2022). Finetuned Language Models Are Zero-Shot Learners. ICLR 2022.",
    "[7] Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35.",
    "[8] Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.",
    "[9] Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. Advances in Neural Information Processing Systems, 35.",
    "[10] Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.",
    "[11] Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. Advances in Neural Information Processing Systems, 36.",
    "[12] Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874.",
    "[13] Sechidis, K. et al. (2011). On the stratification of multi-label data. Proceedings of the 2011 European Conference on Machine Learning, 145–158.",
    "[14] Kuhn, K. (2000). Problems in the automated classification of aviation incident reports. NASA Technical Report.",
    "[15] Robinson, S. et al. (2019). Applying Deep Learning to Aviation Safety Text Classification. IEEE Aerospace Conference Proceedings.",
    "[16] Sun, X. et al. (2023). Text Classification via Large Language Models. Findings of EMNLP 2023.",
    "[17] Wang, Y. et al. (2023). How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources. Advances in Neural Information Processing Systems, 36.",
    "[18] Zhang, X. et al. (2021). BERT-based Classification of Aviation Safety Reports. IEEE Transactions on Intelligent Transportation Systems.",
    "[19] Rose, R. et al. (2023). LLM-Augmented Information Extraction from Aviation Safety Reports. AAAI Workshop on AI for Transportation Safety.",
    "[20] Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.",
    "[21] Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Technical Report.",
    "[22] Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.",
    "[23] Jiang, A. et al. (2023). Mistral 7B. arXiv:2310.06825.",
    "[24] Qwen Team (2024). Qwen3 Technical Report. arXiv:2505.09388.",
    "[25] DeepSeek AI (2025). DeepSeek-V3 Technical Report. arXiv:2412.19437.",
    "[26] Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems, 30.",
    "[27] Zhang, M.-L. & Zhou, Z.-H. (2014). A Review on Multi-Label Learning Algorithms. IEEE Transactions on Knowledge and Data Engineering, 26(8), 1819–1837.",
    "[28] NASA ASRS (2024). Aviation Safety Reporting System Database Online. https://asrs.arc.nasa.gov/",
    "[29] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825\u20132830.",
    "[30] Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. Proceedings of the 29th Symposium on Operating Systems Principles, 611\u2013626.",
    "[31] Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019.",
    "[32] Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
    "[33] He, P. et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR 2021.",
    "[34] Su, J. et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 568, 127063.",
  ];

  for (const r of refs) {
    charCount += r.length;
    p.push(new Paragraph({
      children: [new TextRun({ text: r, font: FONT, size: FONT_SIZE_HF })],
      spacing: { before: 40, after: 40, line: LINE_SPACING },
      indent: { left: 480, hanging: 480 },
    }));
  }

  return p;
}

// ─── APPENDICES ─────────────────────────────────────────────────────────────

function buildAppendices() {
  const p = [];
  p.push(heading1(t("ПРИЛОЖЕНИЯ")));

  // Appendix A: Prompt Templates
  p.push(heading2(t("Приложение А. Шаблони на промптове")));
  p.push(heading3(t("А.1. Основен системен промпт (Basic)")));
  const basicPrompt = `You are an aviation safety analyst classifying ASRS incident reports. For each report, identify ALL applicable anomaly categories from the list below. A report can belong to multiple categories. Return ONLY a JSON array of matching category names, nothing else.

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
- Inflight Event/Encounter`;
  charCount += basicPrompt.length;
  p.push(new Paragraph({
    children: [new TextRun({ text: basicPrompt, font: "Courier New", size: 18 })],
    spacing: { before: 60, after: 60 },
    indent: { left: 360 },
  }));

  p.push(heading3(t("А.2. Таксономично обогатен системен промпт (Taxonomy)")));
  const taxPrompt = `You are an expert aviation safety analyst trained in NASA ASRS report classification.

Classify the following aviation safety report into one or more anomaly categories.
Output ONLY a JSON array of matching category names, nothing else.

Categories (with official NASA ASRS taxonomy subcategories):

1. Aircraft Equipment Problem: Aircraft system malfunction or failure.
   Subcategories: Critical, Less Severe.

2. Airspace Violation: Unauthorized entry or operation in controlled/restricted airspace.

3. ATC Issue: Problems involving air traffic control services, instructions, or communications.

4. Conflict: Loss of separation or near collision.
   Subcategories: NMAC, Airborne Conflict, Ground Conflict.

5. Deviation - Altitude: Departure from assigned altitude.
   Subcategories: Excursion from Assigned Altitude, Crossing Restriction Not Met, Undershoot, Overshoot.

6. Deviation - Procedural: Departure from established procedures, clearances, regulations, or policies.
   Note: This is the broadest category (~65% of reports).

7. Deviation - Speed: Departure from assigned or appropriate speed.

8. Deviation - Track/Heading: Departure from assigned or intended track or heading.

9. Flight Deck/Cabin Event: Events in the flight deck or cabin.
   Subcategories: Illness/Injury, Passenger Misconduct, Smoke/Fire/Fumes/Odor.

10. Ground Event/Encounter: Events occurring ON the ground involving equipment or objects.

11. Ground Excursion: Aircraft LEAVING the intended surface.

12. Ground Incursion: Unauthorized ENTRY onto a surface.

13. Inflight Event/Encounter: Events occurring IN THE AIR.

IMPORTANT distinctions:
- Ground Excursion = aircraft LEAVING the intended surface vs Ground Incursion = unauthorized ENTRY
- Ground Event/Encounter = events ON the ground vs Inflight Event/Encounter = events IN THE AIR
- Deviation - Procedural is very broad -- when in doubt, include it

A report can belong to multiple categories. Be precise -- avoid over-predicting.`;
  charCount += taxPrompt.length;
  p.push(new Paragraph({
    children: [new TextRun({ text: taxPrompt, font: "Courier New", size: 18 })],
    spacing: { before: 60, after: 60 },
    indent: { left: 360 },
  }));

  // Appendix B: Sample Reports
  p.push(heading2(t("Приложение Б. Примерни ASRS доклади")));
  const samples = [
    {
      acn: "1333851",
      labels: "Aircraft Equipment Problem",
      text: "Part 91 flight. Aircraft just released for flight after a heavy c-check. During first leg of the ferry; during ILS approach to landing; right main landing gear did not extend during normal extension. Using QRH procedures for manual extension; all main gear extended to down and locked. As a side note; this airport requires the landing gear to be extended while out over the water well in advance of when the crew would normally extend the gear. As a flight test crew; we were very familiar with the procedure and time required to manually extend the gear. Using the QRH manual extension system; we had 3 down and locked and all checklist completed well in advance of the required stable call. Uneventful landing.",
    },
    {
      acn: "581639",
      labels: "Conflict",
      text: "RETURNING FROM THE FRENCH VALLEY ARPT TO TOA USING FLT FOLLOWING FROM SOCAL APCH. JUST AS I WAS RELEASED TO BEGIN DSCNT BELOW 3000 FT; SOCAL PROVIDED A TA; A CHEROKEE 3 MI AT 9-10 O'CLOCK; 1900 FT DSNDING. A LITTLE LATER; A SECOND ADVISORY CAME. THIS TIME; SOCAL RPTED THAT THE PLANE HAD ME IN SIGHT. I REPLIED TO BOTH RPTS WITH LOOKING; NOT IN SIGHT. THEN SOCAL SAID LET'S DECLARE A LITTLE EMER HERE; GIVE ME MAX CLB FOR SPACING. I COMPLIED; AND ALMOST IMMEDIATELY SAW THE TFC WELL CLR TO MY L AND LOWER.",
    },
    {
      acn: "1117236",
      labels: "Ground Event/Encounter, Ground Incursion",
      text: "Truck towing a piece of farm implement drove onto the taxiway and past the runway hold bars without stopping and or looking and at the edge of the runway turned into the field to unload the piece of equipment. Vehicle had no lights; flags; or VHF radio with numerous aircraft in the traffic pattern. No effort to remove F.O.D. from taxiway/runway from the dirt on vehicle tires. The piece of farm machinery; a plow; is now next to Runway 16/34; unmarked; not NOTAMed; unlighted. Yolo County Airport is unfenced; non-gated; poor to no markings identifying either taxiways or active runways.",
    },
    {
      acn: "693160",
      labels: "Aircraft Equipment Problem, Deviation - Procedural, Deviation - Track/Heading",
      text: "DURING PREFLT WE PLANNED AN E SIDE DEP. ON TAXI OUT WE WERE ASSIGNED A W SIDE RWY AT DFW. I RELOADED THE RWY AND DEP. DOING SO I MUST HAVE MISS LOADED THE TRANSITION. I CLOSED THE DISCONTINUITY IN THE FLT PLAN AND DIDNT NOTICE THAT IT WENT FROM GRABE INTXN TO EOS. WE GOT TO CRUISE ALT AND WERE LOOKING AT THE NEXT WAYPOINT AFTER GRABE AND IT DIDN'T COINCIDE WITH OUR PLAN. CTR CALLED AND CLRED US DIRECT TO OKM. CAUSE OF PROBLEM WAS MISS LOAD OF FMS; NOT RECHKING LOAD.",
    },
    {
      acn: "1652671",
      labels: "ATC Issue, Conflict, Deviation - Procedural",
      text: "Denver TRACON has opted to implement new procedures to fix overshoots on the 16 Runway finals. At any given time one of the 16 finals is called the 'high' runway and is told to conduct ILS approaches while the other is the 'low' runway and is told to conduct visual approaches. However here at Denver we also have RNAV ZULU Approaches. It was decided that I was going to go from being the 'low' runway to being the 'high' runway. Unfortunately this was never coordinated with me. As I began to run the final I noticed an RNAV ZULU approach within 3 miles of the aircraft I had cleared for the ILS. I issued the visual approach clearance but at that point it was already a loss of separation.",
    },
    {
      acn: "580113",
      labels: "Deviation - Procedural, Ground Event/Encounter, Ground Excursion",
      text: "PREFLT AND START-UP WERE COMPLETED AS NORMAL. AS TAXI WAS STARTED; I TESTED THE L AND R RUDDER PEDALS; THEN THE BRAKES. THE STUDENT THEN DID THE SAME. ONCE REACHING THE INTXN OF TXWY I AND TXWY G; THE STUDENT RECEIVED A TAXI CLRNC TO RWY 28R VIA TXWY H AND TXWY A. UPON REACHING THE INTXN OF TXWY H AND TXWY A; THE STUDENT APPLIED L RUDDER AND THE ACFT CONTINUED STRAIGHT AHEAD. I REACTED BY PULLING THE REMAINING PWR AND APPLYING HVY BRAKING. THE ACFT WAS UNABLE TO BE STOPPED ON THE TXWY AND DID NOT COME TO A COMPLETE STOP UNTIL IN THE DIRT AT THE E END OF THE TXWY.",
    },
    {
      acn: "695364",
      labels: "Airspace Violation, Deviation - Procedural",
      text: "A YOUNG MAN RAN AWAY FROM A PVT FACILITY IN THE DESERT. I HAD AN AIRPLANE AND ONE OF THE DIRECTORS NEEDED A WAY TO SEARCH FOR THE YOUNG MAN. I CALLED FSS ON MY CELL AND GOT MY HOME FSS AT DAYTON. THERE WERE NO RESTRS TO FLT NOTED. ON MY WAY TO THE PLACE OF SEARCH; I FLEW NEAR A BUNCH OF BUILDINGS. I THEN REMEMBERED A TFR BEING IN THAT AREA IN THE LAST SEVERAL YRS. I FLEW S IMMEDIATELY AS SOON AS MY MIND REALIZED THAT I MIGHT BE NEAR A TFR. I CALLED CEDAR CITY FSS AND THEY TOLD ME IT WAS NOW A PERMANENT FLT RESTR.",
    },
    {
      acn: "2004094",
      labels: "Conflict, Deviation - Procedural",
      text: "Landing on Runway XX at ZZZ; Aircraft Y took off on the crossing Runway YY. Looking at the recording we made a 10 mile; 5 mile; and short final call. While the Aircraft Y did make a take-off call; they took off while we were short final and we missed the aircraft by 500 ft. horizontally. Used the term did not see you on the scope and based upon the conversation they were not listening to the CTAF frequency.",
    },
  ];
  for (const s of samples) {
    p.push(bodyNoIndent([
      bold(t(`ACN ${s.acn}`)),
      run(t(` — Категории: ${s.labels}`)),
    ]));
    charCount += s.text.length;
    p.push(new Paragraph({
      children: [new TextRun({ text: s.text, font: FONT, size: 20, italics: true })],
      spacing: { before: 40, after: 100, line: 300 },
      indent: { left: 360 },
    }));
  }

  // Appendix C: All Experiments Table
  p.push(heading2(t("Приложение В. Обобщение на всички експерименти")));

  const tNumApp = nextTable();
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumApp}. Пълен списък на проведените експерименти`))]));
  p.push(makeDataTable(
    ["#", "Модел", "Подход", "Macro-F1", "Micro-F1", "AUC", "Цена"],
    [
      ["1", "Classic ML 32K", "TF-IDF+XGB", "0.691", "0.746", "0.932", "$0"],
      ["2", "Classic ML 164K", "TF-IDF+XGB", "0.678", "0.739", "0.942", "$0.64"],
      ["3", "Classic ML Tuned", "TF-IDF+XGB", "0.693", "0.745", "0.932", "$3.30"],
      ["4", "Qwen3-8B", "ZS basic", "0.459", "0.473", "0.727", "$0.35"],
      ["5", "Qwen3-8B", "FS basic", "0.453", "0.468", "0.704", "$0.46"],
      ["6", "Qwen3-8B", "Fine-tuned", "0.510", "0.632", "0.700", "$10.83"],
      ["7", "Qwen3-8B", "ZS taxonomy", "0.499", "0.605", "0.701", "$0.33"],
      ["8", "Qwen3-8B", "FS taxonomy", "0.526", "0.544", "0.706", "$0.45"],
      ["9", "Qwen3-8B", "FS tax+think", "0.533", "0.556", "0.705", "$6.67"],
      ["10", "Mistral Large 3", "ZS taxonomy", "0.658", "0.712", "0.793", "$0"],
      ["11", "Mistral Large 3", "FS taxonomy", "0.640", "0.686", "0.793", "$0"],
      ["12", "Ministral 8B", "ZS basic", "0.491", "0.543", "0.744", "$0.25"],
      ["13", "Ministral 8B", "FS basic", "0.540", "0.536", "0.746", "$0.41"],
      ["14", "Ministral 8B", "Fine-tuned", "0.489", "0.542", "0.744", "$10.95"],
      ["15", "DeepSeek V3.2", "ZS taxonomy", "0.623", "0.693", "0.746", "$1.39"],
      ["16", "DeepSeek V3.2", "ZS tax+think", "0.681", "0.723", "0.810", "$6.73"],
      ["17", "CML sub-48", "TF-IDF+XGB", "0.510", "0.600", "0.934", "$3.03"],
      ["18", "CML tuned sub", "TF-IDF+XGB", "0.510", "0.600", "0.934", "$3.30"],
      ["19", "Mistral Lg sub", "ZS taxonomy", "0.449", "0.494", "0.744", "paid"],
      ["20", "DeepSeek sub", "ZS taxonomy", "0.422", "0.456", "0.708", "$1.92"],
      ["21", "DeepSeek sub", "ZS tax+think", "0.419", "0.466", "0.690", "$5.24"],
      ["22", "Qwen3-8B sub", "ZS taxonomy", "0.235", "0.304", "0.629", "$0.40"],
    ],
  ));

  // Appendix D: Glossary
  p.push(heading2(t("Приложение Г. Речник на термините")));
  const glossary = [
    ["ASRS", "Aviation Safety Reporting System \u2014 система за доброволно докладване на авиационни инциденти, управлявана от NASA"],
    ["TF-IDF", "Term Frequency \u2013 Inverse Document Frequency \u2014 метод за извличане на числови признаци от текст, претеглящ термините по тяхната специфичност"],
    ["XGBoost", "Extreme Gradient Boosting \u2014 ансамблов алгоритъм за класификация, базиран на градиентно усилване на дървета за решение"],
    ["LLM", "Large Language Model \u2014 голям езиков модел с милиарди параметъра, предобучен на огромни текстови корпуси"],
    ["QLoRA", "Quantized Low-Rank Adaptation \u2014 метод за фино настройване на LLM с 4-битова квантизация и нискорангови адаптери"],
    ["LoRA", "Low-Rank Adaptation \u2014 метод за ефективно фино настройване чрез добавяне на нискорангови матрици към избрани слоеве"],
    ["MoE", "Mixture of Experts \u2014 архитектура, при която се активира само подмножество от параметрите при всяко извикване"],
    ["NF4", "Normal Float 4-bit \u2014 формат за 4-битова квантизация, оптимизиран за нормално разпределени тегла"],
    ["vLLM", "Високопроизводителна библиотека за инференция на LLM с PagedAttention и continuous batching"],
    ["Macro-F1", "Макро-усреднен F1-Score \u2014 аритметична средна на F1 за всяка категория (еднаква тежест)"],
    ["Micro-F1", "Микро-усреднен F1-Score \u2014 глобално изчислен F1, претеглен по честотата на категориите"],
    ["ROC-AUC", "Area Under the Receiver Operating Characteristic Curve \u2014 метрика за способността на модела да ранжира положителни над отрицателни примери"],
    ["Binary Relevance", "Подход за многоетикетна класификация с независим бинарен класификатор за всяка категория"],
    ["Zero-Shot", "Нулево обучение \u2014 класификация без примери, само чрез текстова инструкция"],
    ["Few-Shot", "Обучение с малко примери \u2014 класификация с 2\u20133 примера за категория в промпта"],
    ["Batch API", "API за асинхронна пакетна обработка на множество заявки без ограничение на честотата"],
    ["CoT", "Chain-of-Thought \u2014 техника за генериране на междинни стъпки на разсъждение преди крайния отговор"],
    ["BPE", "Byte Pair Encoding \u2014 алгоритъм за подсловна токенизация, използван от повечето LLM"],
    ["RoPE", "Rotary Position Embedding \u2014 метод за кодиране на позицията чрез ротация на векторното пространство"],
    ["ACN", "Accession Number \u2014 уникален идентификатор на ASRS доклад"],
  ];
  const tNumGloss = nextTable();
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumGloss}. Речник на основните термини и съкращения`))]));
  p.push(makeDataTable(
    ["Термин", "Описание"],
    glossary,
  ));
  p.push(emptyLine());

  // Appendix E: Compute Cost Log
  p.push(heading2(t("Приложение Д. Дневник на изчислителните разходи")));

  const tNumComp = nextTable();
  p.push(bodyNoIndent([bold(t(`Таблица ${tNumComp}. Подробен дневник на изчислителните разходи`))]));
  p.push(makeDataTable(
    ["Експеримент", "GPU/API", "Време", "Цена"],
    [
      ["Classic ML (XGBoost)", "CPU (локален)", "~55 min", "$0"],
      ["ZS Qwen3-8B", "L4 (Modal)", "~26 min", "~$0.35"],
      ["FS Qwen3-8B", "L4 (Modal)", "~34 min", "~$0.46"],
      ["QLoRA обучение", "A100 (Modal)", "~227 min", "~$10.56"],
      ["Fine-tuned инференция", "L4 (Modal)", "~20 min", "~$0.27"],
      ["ZS Mistral Large 3", "Batch API", "~5 min", "$0"],
      ["FS Mistral Large 3", "Batch API", "~4 min", "$0"],
      ["ZS taxonomy Qwen3", "L4 (Modal)", "~24 min", "~$0.33"],
      ["FS taxonomy Qwen3", "L4 (Modal)", "~34 min", "~$0.45"],
      ["FS tax+thinking Qwen3", "A100 (Modal)", "~144 min", "~$6.67"],
      ["ZS DeepSeek V3.2", "DeepInfra API", "~7 min", "~$1.39"],
      ["ZS+think DeepSeek", "DeepInfra API", "~291 min", "~$6.73"],
      ["CML subcategory", "32-core CPU", "~142 min", "~$3.03"],
      ["ZS sub Mistral Large", "Real-time API", "~119 min", "paid"],
      ["ZS sub Qwen3", "L4 (Modal)", "~30 min", "~$0.40"],
      ["ZS sub DeepSeek", "DeepInfra API", "~8 min", "~$1.92"],
      ["ZS+think sub DeepSeek", "DeepInfra API", "~545 min", "~$5.24"],
      ["CML tuning Phase 3", "32-core CPU", "~154 min", "~$3.30"],
      ["ОБЩО", "", "", "~$53"],
    ],
  ));

  return p;
}

// ─── DOCUMENT ASSEMBLY ──────────────────────────────────────────────────────

function buildDocument() {
  console.log("Building thesis document...");

  // Gather all content
  const titlePage = buildTitlePage();
  const annotation = buildAnnotation();
  const toc = buildTOC();
  const intro = buildIntroduction();
  const ch1 = buildChapter1();
  const ch2 = buildChapter2();
  const ch3 = buildChapter3();
  const ch4 = buildChapter4();
  const conclusion = buildConclusion();
  const bib = buildBibliography();
  const appendices = buildAppendices();

  const allContent = [
    ...titlePage, ...annotation, ...toc,
    ...intro, ...ch1, ...ch2, ...ch3, ...ch4,
    ...conclusion, ...bib, ...appendices,
  ];

  console.log(`Total character count (approx): ${charCount.toLocaleString()}`);
  if (charCount < 117000) {
    console.warn(`WARNING: Character count ${charCount} is below 117,000 target.`);
    console.warn("Consider expanding Literature Review and Discussion chapters.");
  } else {
    console.log(`Character count meets 117,000+ target.`);
  }

  // Section 1: Title page (no page numbers)
  // Section 2: Body (page numbers from 2)
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: FONT, size: FONT_SIZE_HF },
          paragraph: {
            spacing: { line: LINE_SPACING },
            alignment: AlignmentType.JUSTIFIED,
          },
        },
        heading1: {
          run: { font: FONT, size: HEADING1_SIZE, bold: true },
          paragraph: {
            spacing: { before: 240, after: 120, line: LINE_SPACING },
            alignment: AlignmentType.CENTER,
          },
        },
        heading2: {
          run: { font: FONT, size: HEADING2_SIZE, bold: true },
          paragraph: {
            spacing: { before: 240, after: 120, line: LINE_SPACING },
          },
        },
        heading3: {
          run: { font: FONT, size: HEADING3_SIZE, bold: true },
          paragraph: {
            spacing: { before: 200, after: 100, line: LINE_SPACING },
          },
        },
      },
    },
    features: {
      updateFields: true,
    },
    sections: [
      // Section 1: Title page (no headers/footers)
      {
        properties: {
          page: {
            margin: MARGINS,
            pageNumbers: { start: 0 },
          },
        },
        children: [...titlePage],
      },
      // Section 2: Body with page numbers
      {
        properties: {
          page: {
            margin: MARGINS,
            pageNumbers: { start: 2 },
          },
          type: SectionType.NEXT_PAGE,
        },
        headers: {
          default: new Header({
            children: [new Paragraph({ children: [] })],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                children: [
                  new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 }),
                ],
                alignment: AlignmentType.CENTER,
              }),
            ],
          }),
        },
        children: [
          ...annotation, ...toc,
          ...intro, ...ch1, ...ch2, ...ch3, ...ch4,
          ...conclusion, ...bib, ...appendices,
        ],
      },
    ],
  });

  return doc;
}

async function main() {
  const doc = buildDocument();
  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.join(ROOT, "thesis.docx");
  fs.writeFileSync(outputPath, buffer);
  console.log(`\nThesis written to: ${outputPath}`);
  console.log(`File size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Figures: ${figureCounter}`);
  console.log(`Tables: ${tableCounter}`);
}

main().catch(err => {
  console.error("Error:", err);
  process.exit(1);
});
