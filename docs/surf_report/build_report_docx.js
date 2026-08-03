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
  Math: OMath, MathRun, MathSubScript, MathSuperScript, MathSubSuperScript,
} = require("docx");

// ---- equations rendered as native Word (OMML) math ----
const mr = (t) => new MathRun(t);
const msub = (b, s) => new MathSubScript({ children: [mr(b)], subScript: [mr(s)] });
const msup = (b, s) => new MathSuperScript({ children: [mr(b)], superScript: [mr(s)] });
const msubsup = (b, s, p) => new MathSubSuperScript({ children: [mr(b)], subScript: [mr(s)], superScript: [mr(p)] });
const EQMAP = {
  eq_Ci: () => [msub("C", "i"), mr(" = ρ·"), msub("c", "p"), mr("·"), msub("V", "i")],
  eq_Gij: () => [msub("G", "ij")],
  eq_ode: () => [mr("C(T) Ṫ = −L(T) T + P + εσA("), msubsup("T", "env", "4"), mr(" − "), msup("T", "4"), mr(")")],
  eq_yGu: () => [mr("y = G u")],
  eq_pinv: () => [mr("u = "), msup("G", "+"), mr(" Δy")],
  eq_ulim: () => [mr("0 ≤ u ≤ "), msub("u", "max")],
  eq_uge: () => [mr("u ≥ 0")],
  eq_ulem: () => [mr("u ≤ "), msub("u", "max")],
  eq_ssfull: () => [mr("ẋ = A x + B u,  y = C x")],
  eq_eig: () => [mr("(L + diag("), msub("G", "rad"), mr(")) φ = λ C φ")],
  eq_orthon: () => [msup("Φ", "T"), mr(" C Φ = I")],
  eq_modal: () => [msub("A", "mod"), mr(" = −diag(λ),  "), msub("B", "mod"), mr(" = "), msup("Φ", "T"),
                   mr(" F,  "), msub("C", "out"), mr(" = S Φ")],
  eq_uKx: () => [mr("u = −K x")],
  eq_cost: () => [mr("∫("), msup("y", "T"), mr(" y + ρ "), msup("u", "T"), mr(" u) dt")],
};

const HERE = __dirname;
const FONT = "Verdana";
const SZ = 16;        // 8pt body
const SZ_H2 = 18;     // 9pt section heading
const SZ_TITLE = 20;  // 10pt title
const single = { before: 0, after: 0, line: 240, lineRule: "auto" };
const USABLE_DXA = 9360; // 12240 - 2*1440 margins

const md = fs.readFileSync(path.join(HERE, "interim_report_2.md"), "utf8");
const lines = md.split(/\r?\n/);

// ---- inline $eq$ math + **bold** / *italic* / `code` -> (TextRun|Math)[] ----
function inlineRuns(text, base = {}) {
  const out = [];
  const parts = text.split(/\$([a-zA-Z_][a-zA-Z0-9_]*)\$/); // odd indices = equation keys
  parts.forEach((seg, idx) => {
    if (idx % 2 === 1) {
      const f = EQMAP[seg];
      if (f) { out.push(new OMath({ children: f() })); }
      else out.push(new TextRun({ text: `$${seg}$`, font: FONT, size: base.size || SZ, ...base }));
      return;
    }
    const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`)/g;
    let last = 0, m;
    const push = (t, opts) => { if (t) out.push(new TextRun({ text: t, font: FONT, size: base.size || SZ, ...base, ...opts })); };
    while ((m = re.exec(seg)) !== null) {
      push(seg.slice(last, m.index));
      if (m[2] !== undefined) push(m[2], { bold: true });
      else if (m[3] !== undefined) push(m[3], { italics: true });
      else if (m[4] !== undefined) push(m[4], {}); // code -> plain Verdana
      last = re.lastIndex;
    }
    push(seg.slice(last));
  });
  return out.length ? out : [new TextRun({ text: "", font: FONT, size: base.size || SZ, ...base })];
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
  const out = path.join(HERE, process.argv[2] || "interim_report_2.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${(buf.length/1024).toFixed(0)} KB)`);
});
