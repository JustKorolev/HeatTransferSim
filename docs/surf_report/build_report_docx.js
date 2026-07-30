// Builds interim_report_2.docx from interim_report_2.md, formatted the same way
// as abstract.docx (Verdana 8pt, left-justified, single spacing, 0pt before/after,
// bold sentence-case title, plain name, italic mentor). Parses a constrained
// subset of Markdown (headings, paragraphs, **bold**/*italic*/`code`, bullets,
// numbered items, a pipe table, and ![](img)) so the .md stays the source of truth.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle,
} = require("docx");

const HERE = __dirname;
const FONT = "Verdana";
const SZ = 16;        // 8pt body
const SZ_H2 = 18;     // 9pt section heading
const SZ_TITLE = 20;  // 10pt title
const single = { before: 0, after: 0, line: 240, lineRule: "auto" };
const USABLE_DXA = 9360; // 12240 - 2*1440 margins

const md = fs.readFileSync(path.join(HERE, "interim_report_2.md"), "utf8");
const lines = md.split(/\r?\n/);

// ---- inline **bold** / *italic* / `code` -> TextRun[] ----
function inlineRuns(text, base = {}) {
  const runs = [];
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`)/g;
  let last = 0, m;
  const push = (t, opts) => { if (t) runs.push(new TextRun({ text: t, font: FONT, size: base.size || SZ, ...base, ...opts })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    if (m[2] !== undefined) push(m[2], { bold: true });
    else if (m[3] !== undefined) push(m[3], { italics: true });
    else if (m[4] !== undefined) push(m[4], {}); // code -> plain Verdana
    last = re.lastIndex;
  }
  push(text.slice(last));
  return runs.length ? runs : [new TextRun({ text: "", font: FONT, size: base.size || SZ, ...base })];
}

const para = (text, opts = {}) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: single, children: inlineRuns(text, opts) });
const blank = () => new Paragraph({ alignment: AlignmentType.LEFT, spacing: single, children: [new TextRun({ text: "", font: FONT, size: SZ })] });

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function image(rel) {
  const buf = fs.readFileSync(path.join(HERE, rel));
  const { w, h } = pngSize(buf);
  const maxW = 560, width = Math.min(maxW, w), height = Math.round(width * h / w);
  return new Paragraph({ alignment: AlignmentType.LEFT, spacing: single,
    children: [new ImageRun({ data: buf, type: "png", transformation: { width, height } })] });
}

function tableFrom(rows) {
  const cols = 4, widths = [3600, 2100, 1900, 1760];
  const border = { style: BorderStyle.SINGLE, size: 4, color: "888888" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const trows = rows.map((cells, ri) => new TableRow({
    children: cells.map((c, ci) => new TableCell({
      width: { size: widths[ci], type: WidthType.DXA }, borders,
      children: [new Paragraph({ alignment: AlignmentType.LEFT, spacing: single,
        children: inlineRuns(c, { bold: ri === 0 }) })],
    })),
  }));
  return new Table({ columnWidths: widths, width: { size: USABLE_DXA, type: WidthType.DXA }, rows: trows });
}

// ---- group lines into blocks separated by blank lines / rules ----
const blocks = [];
let cur = [];
for (const raw of lines) {
  const tr = raw.trim();
  if (tr === "" || tr === "---") { if (cur.length) { blocks.push(cur); cur = []; } continue; }
  cur.push(raw);
}
if (cur.length) blocks.push(cur);

const isMarker = (raw) => raw === raw.trimStart() && (/^- /.test(raw.trim()) || /^\d+\.\s/.test(raw.trim()));

// ---- render each block ----
const children = [];
blocks.forEach((block, bi) => {
  if (bi > 0) children.push(blank());
  const first = block[0].trim();

  if (first.startsWith("# ")) {
    children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: single,
      children: inlineRuns(first.slice(2), { bold: true, size: SZ_TITLE }) }));
    return;
  }
  if (first.startsWith("## ") || first.startsWith("### ")) {
    children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: single,
      children: inlineRuns(first.replace(/^#+\s/, ""), { bold: true, size: SZ_H2 }) }));
    return;
  }
  const img = first.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
  if (img && block.length === 1) { children.push(image(img[1])); return; }

  if (first.startsWith("|")) {
    const rows = [];
    for (const raw of block) {
      const cells = raw.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(s => s.trim());
      if (!cells.every(c => /^:?-{2,}:?$/.test(c))) rows.push(cells);
    }
    children.push(tableFrom(rows));
    return;
  }

  // mixed paragraph / list block: state machine joining wrapped lines
  let paraBuf = [], item = null;
  const flushPara = () => { if (paraBuf.length) { children.push(para(paraBuf.join(" "))); paraBuf = []; } };
  const flushItem = () => { if (item !== null) { children.push(para(item)); item = null; } };
  for (const raw of block) {
    const tr = raw.trim();
    if (isMarker(raw)) {
      flushPara(); flushItem();
      const nm = tr.match(/^(\d+)\.\s+(.*)$/);
      item = nm ? `${nm[1]}.  ${nm[2]}` : `•  ${tr.slice(2)}`;
    } else if (item !== null) {
      item += " " + tr;              // continuation of a list item
    } else {
      paraBuf.push(tr);              // continuation of a paragraph
    }
  }
  flushPara(); flushItem();
});

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 } } }, children }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(HERE, "interim_report_2.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${(buf.length/1024).toFixed(0)} KB)`);
});
