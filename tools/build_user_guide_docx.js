/*
 * Build the user guide as a .docx that Google Docs imports cleanly.
 *
 *     node tools/build_user_guide_docx.js
 *     node tools/build_user_guide_docx.js --in docs/user_guide/USER_GUIDE.md --out docs/user_guide/HeatTransferSim_User_Guide.docx
 *
 * Google Docs has no API here, so the deliverable is a .docx: File > Import (or
 * dropping it in Drive and opening it) produces a real Google Doc with the
 * headings, tables and screenshots intact. Markdown would lose the images, and
 * HTML loses table widths.
 *
 * This is a converter for THIS guide, not a general Markdown engine. It handles
 * exactly what USER_GUIDE.md uses: ATX headings, pipe tables, fenced code,
 * blockquotes, ordered and unordered lists, images, horizontal rules, and
 * inline code/bold/italic/links. Anything else passes through as plain text
 * rather than being silently dropped.
 *
 * Requires the `docx` npm package. It is not vendored here; install it wherever
 * you like and point NODE_PATH at it, e.g.
 *     npm install docx && node tools/build_user_guide_docx.js
 */

const fs = require("fs");
const path = require("path");

const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  ImageRun,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
  LevelFormat,
} = require("docx");

// US Letter, in DXA (1440 per inch). docx-js defaults to A4.
const PAGE_WIDTH = 12240;
const PAGE_HEIGHT = 15840;
const MARGIN = 1080; // 0.75"
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2;

const MONO = "Consolas";
const BODY = "Calibri";
const ACCENT = "C2410C";
const CODE_BG = "F4F4F5";
const HEAD_BG = "EDEDEF";

function args() {
  const out = { in: "docs/user_guide/USER_GUIDE.md", out: "docs/user_guide/HeatTransferSim_User_Guide.docx" };
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i].replace(/^--/, "");
    if (key in out) out[key] = process.argv[i + 1];
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Inline formatting
 * ------------------------------------------------------------------ */
// Split on `code`, **bold**, *italic* and [text](href), in one pass so a
// construct inside another is not re-scanned and mangled.
const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;

function inlineRuns(text, base = {}) {
  const runs = [];
  const pieces = String(text).split(INLINE).filter((p) => p !== "" && p !== undefined);
  for (const piece of pieces) {
    if (piece.startsWith("`") && piece.endsWith("`") && piece.length > 1) {
      runs.push(new TextRun({ ...base, text: piece.slice(1, -1), font: MONO, size: 19, color: "B91C1C" }));
    } else if (piece.startsWith("**") && piece.endsWith("**")) {
      runs.push(new TextRun({ ...base, text: piece.slice(2, -2), bold: true }));
    } else if (piece.startsWith("*") && piece.endsWith("*") && piece.length > 2) {
      runs.push(new TextRun({ ...base, text: piece.slice(1, -1), italics: true }));
    } else if (piece.startsWith("[")) {
      const m = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(piece);
      // Internal anchors (#section) would not resolve after import, so only the
      // link TEXT is kept; external URLs are shown so they stay usable on paper.
      if (m) {
        const external = /^https?:/.test(m[2]);
        runs.push(new TextRun({ ...base, text: m[1], color: external ? "1D4ED8" : undefined }));
        if (external) runs.push(new TextRun({ ...base, text: ` (${m[2]})`, size: 17, color: "6B7280" }));
      } else {
        runs.push(new TextRun({ ...base, text: piece }));
      }
    } else {
      runs.push(new TextRun({ ...base, text: piece }));
    }
  }
  return runs.length ? runs : [new TextRun({ ...base, text: "" })];
}

/* ------------------------------------------------------------------ *
 * Block builders
 * ------------------------------------------------------------------ */
function heading(text, level) {
  const levels = [
    HeadingLevel.TITLE,
    HeadingLevel.HEADING_1,
    HeadingLevel.HEADING_2,
    HeadingLevel.HEADING_3,
    HeadingLevel.HEADING_4,
  ];
  return new Paragraph({
    heading: levels[Math.min(level, levels.length - 1)],
    spacing: { before: level <= 1 ? 360 : 260, after: 120 },
    children: inlineRuns(text),
  });
}

function body(text) {
  return new Paragraph({ spacing: { after: 140, line: 276 }, children: inlineRuns(text) });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "guide-bullets", level },
    spacing: { after: 70, line: 264 },
    children: inlineRuns(text),
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "guide-numbers", level },
    spacing: { after: 70, line: 264 },
    children: inlineRuns(text),
  });
}

function codeBlock(lines) {
  // One paragraph per line: shading is per-paragraph, and "\n" inside a TextRun
  // is not a line break in OOXML.
  return lines.map((line, index) =>
    new Paragraph({
      shading: { type: ShadingType.CLEAR, fill: CODE_BG, color: "auto" },
      spacing: { before: index === 0 ? 100 : 0, after: index === lines.length - 1 ? 160 : 0 },
      indent: { left: 240, right: 240 },
      children: [new TextRun({ text: line || " ", font: MONO, size: 19 })],
    })
  );
}

function quote(lines) {
  return lines.map((line, index) =>
    new Paragraph({
      indent: { left: 360 },
      spacing: { before: index === 0 ? 120 : 0, after: index === lines.length - 1 ? 160 : 60 },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 12 } },
      children: inlineRuns(line, { color: "374151" }),
    })
  );
}

function rule() {
  return new Paragraph({
    spacing: { before: 160, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "D4D4D8", space: 4 } },
    children: [new TextRun("")],
  });
}

function table(rows) {
  const columns = rows[0].length;
  // Both the table's columnWidths and every cell's width must be set, in DXA;
  // percentages break on import into Google Docs.
  const widths = Array.from({ length: columns }, () => Math.floor(CONTENT_WIDTH / columns));
  widths[0] = CONTENT_WIDTH - widths.slice(1).reduce((a, b) => a + b, 0);

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    rows: rows.map((cells, rowIndex) =>
      new TableRow({
        tableHeader: rowIndex === 0,
        children: cells.map((cell, columnIndex) =>
          new TableCell({
            width: { size: widths[columnIndex], type: WidthType.DXA },
            shading: rowIndex === 0
              ? { type: ShadingType.CLEAR, fill: HEAD_BG, color: "auto" }
              : undefined,
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: [
              new Paragraph({
                spacing: { after: 0 },
                children: inlineRuns(cell, rowIndex === 0 ? { bold: true } : {}),
              }),
            ],
          })
        ),
      })
    ),
  });
}

function image(altText, relativePath, baseDir) {
  const file = path.resolve(baseDir, relativePath);
  if (!fs.existsSync(file)) {
    console.warn(`  ! missing image: ${relativePath}`);
    return [body(`[missing image: ${relativePath}]`)];
  }
  const data = fs.readFileSync(file);
  const size = pngSize(data);
  // Fit the content width, preserving aspect, and never upscale past it.
  const maxWidthPx = 620;
  const scale = size.width > maxWidthPx ? maxWidthPx / size.width : 1;
  const out = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [
        new ImageRun({
          type: "png",
          data,
          transformation: {
            width: Math.round(size.width * scale),
            height: Math.round(size.height * scale),
          },
        }),
      ],
    }),
  ];
  if (altText) {
    out.push(
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: altText, italics: true, size: 18, color: "6B7280" })],
      })
    );
  }
  return out;
}

/** Width/height from a PNG's IHDR, so images keep their aspect ratio. */
function pngSize(buffer) {
  if (buffer.length < 24 || buffer.readUInt32BE(0) !== 0x89504e47) {
    return { width: 600, height: 400 };
  }
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

/* ------------------------------------------------------------------ *
 * Markdown walk
 * ------------------------------------------------------------------ */
function convert(markdown, baseDir) {
  const lines = markdown.split(/\r?\n/);
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed === "") { i++; continue; }

    if (/^```/.test(trimmed)) {
      const buffer = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) buffer.push(lines[i++]);
      i++;
      out.push(...codeBlock(buffer));
      continue;
    }

    if (/^(---|\*\*\*|___)\s*$/.test(trimmed)) { out.push(rule()); i++; continue; }

    const headingMatch = /^(#{1,5})\s+(.*)$/.exec(trimmed);
    if (headingMatch) {
      out.push(heading(headingMatch[2], headingMatch[1].length - 1));
      i++;
      continue;
    }

    const imageMatch = /^!\[([^\]]*)\]\(([^)]+)\)\s*$/.exec(trimmed);
    if (imageMatch) {
      out.push(...image(imageMatch[1], imageMatch[2], baseDir));
      i++;
      continue;
    }

    if (trimmed.startsWith(">")) {
      const buffer = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        buffer.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      out.push(...quote(buffer.filter((l) => l !== "")));
      continue;
    }

    // A pipe table: header row, a separator of dashes, then body rows.
    if (trimmed.startsWith("|") && i + 1 < lines.length && /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())) {
      const rows = [];
      rows.push(splitRow(lines[i]));
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(splitRow(lines[i++]));
      out.push(table(rows));
      out.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun("")] }));
      continue;
    }

    const orderedMatch = /^(\s*)(\d+)\.\s+(.*)$/.exec(line);
    if (orderedMatch) {
      out.push(numbered(orderedMatch[3], Math.min(2, Math.floor(orderedMatch[1].length / 3))));
      i++;
      continue;
    }

    const bulletMatch = /^(\s*)[-*]\s+(.*)$/.exec(line);
    if (bulletMatch) {
      out.push(bullet(bulletMatch[2], Math.min(2, Math.floor(bulletMatch[1].length / 2))));
      i++;
      continue;
    }

    // A plain paragraph: join its continuation lines so it wraps naturally.
    const buffer = [trimmed];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,5}\s|```|>|\||!\[|\s*[-*]\s|\s*\d+\.\s|---)/.test(lines[i])
    ) {
      buffer.push(lines[i].trim());
      i++;
    }
    out.push(body(buffer.join(" ")));
  }
  return out;
}

function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

/* ------------------------------------------------------------------ *
 * Main
 * ------------------------------------------------------------------ */
async function main() {
  const options = args();
  const inputPath = path.resolve(options.in);
  const outputPath = path.resolve(options.out);
  const markdown = fs.readFileSync(inputPath, "utf8");

  console.log(`reading  ${options.in}`);
  const children = convert(markdown, path.dirname(inputPath));
  console.log(`built    ${children.length} block(s)`);

  const doc = new Document({
    creator: "HeatTransferSim",
    title: "HeatTransferSim User Guide",
    description: "Guide to building, simulating and exporting a thermal controller.",
    numbering: {
      config: [
        {
          reference: "guide-bullets",
          levels: [0, 1, 2].map((level) => ({
            level,
            format: LevelFormat.BULLET,
            text: ["•", "◦", "▪"][level],
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420 + level * 360, hanging: 240 } } },
          })),
        },
        {
          reference: "guide-numbers",
          levels: [0, 1, 2].map((level) => ({
            level,
            format: [LevelFormat.DECIMAL, LevelFormat.LOWER_LETTER, LevelFormat.LOWER_ROMAN][level],
            text: [`%1.`, `%2.`, `%3.`][level],
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420 + level * 360, hanging: 240 } } },
          })),
        },
      ],
    },
    styles: {
      default: {
        document: { run: { font: BODY, size: 21 } },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
            margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
          },
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({ text: "HeatTransferSim User Guide  —  ", size: 17, color: "9CA3AF" }),
                  new TextRun({ children: [PageNumber.CURRENT], size: 17, color: "9CA3AF" }),
                ],
              }),
            ],
          }),
        },
        children,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buffer);
  console.log(`wrote    ${options.out}  (${Math.round(buffer.length / 1024)} KB)`);
  console.log("\nTo get it into Google Docs: upload to Drive, then open it and");
  console.log("File > Save as Google Docs (or Drive settings > Convert uploads).");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
