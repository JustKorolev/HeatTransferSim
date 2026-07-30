// Builds abstract.docx to the SFP formatting spec:
// Verdana 8pt, left-justified, bold sentence-case title, plain name,
// italic "Mentor:" line, single line spacing, 0pt before/after.
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType } = require("docx");

const FONT = "Verdana";
const SIZE = 16; // half-points => 8pt
const single = { before: 0, after: 0, line: 240, lineRule: "auto" };

const TITLE =
  "A reduced-order thermal model and multi-input controller for the HISPEC cryostat";
const NAME = "Andrey Korolev";
const MENTOR = "Mentor: Jake Zimmer";
const BODY =
  "Cryogenic astronomical instruments such as the HISPEC spectrograph must hold " +
  "large optical assemblies at a stable operating temperature near 55 K, yet their " +
  "thermal behavior is set by hundreds of thousands of coupled conduction and " +
  "radiation pathways, far too many states to simulate in real time or to " +
  "design a controller against. This project develops an end-to-end pipeline that " +
  "turns a CAD assembly into a lumped thermal-network model and reduces it to a " +
  "compact form suitable for on-board control. From the instrument geometry an " +
  "octree mesh builds a 469,315-node thermal graph; the linearized dynamics are " +
  "projected onto their 120 slowest physical modes and then reduced by balanced " +
  "truncation to a 30-state model that keeps only the jointly controllable and " +
  "observable dynamics. On this reduced model I design a multi-input " +
  "linear-quadratic-regulator controller with a static state estimator and a " +
  "regularized steady-state feedforward for its 33 heaters and 28 controlled " +
  "sensors. The 30-state model reproduces the full plant's heater-to-sensor step " +
  "response to within about 0.15%, small enough to design against while remaining " +
  "light enough to run on a microcontroller. Remaining work adds surface-to-surface " +
  "radiation and hardware deployment.";

const run = (text, opts = {}) =>
  new TextRun({ text, font: FONT, size: SIZE, ...opts });

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: SIZE } },
    },
  },
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: single,
          children: [run(TITLE, { bold: true })],
        }),
        new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: single,
          children: [run(NAME)],
        }),
        new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: single,
          children: [run(MENTOR, { italics: true })],
        }),
        new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: single,
          children: [run("")],
        }),
        new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: single,
          children: [run(BODY)],
        }),
      ],
    },
  ],
});

const out = path.join(__dirname, "abstract.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  const words = BODY.split(/\s+/).filter(Boolean).length;
  console.log("wrote", out, "| body word count:", words);
});
