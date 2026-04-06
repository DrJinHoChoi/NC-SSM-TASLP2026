const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, LevelFormat
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

// Helper: header cell
function hCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "1B3A5C", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Malgun Gothic", size: 18 })] })]
  });
}

// Helper: body cell
function bCell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: cellMargins,
    shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, font: "Malgun Gothic", size: 18, bold: !!opts.bold, color: opts.color || "000000" })]
    })]
  });
}

// Section heading
function sectionHeading(text) {
  return new Paragraph({
    spacing: { before: 360, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1B3A5C", space: 4 } },
    children: [new TextRun({ text, bold: true, font: "Malgun Gothic", size: 26, color: "1B3A5C" })]
  });
}

// Sub heading
function subHeading(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Malgun Gothic", size: 22, color: "2E75B6" })]
  });
}

// Bullet
function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, font: "Malgun Gothic", size: 19 })]
  });
}

// Sub-bullet (dash)
function subBullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 1 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Malgun Gothic", size: 18, color: "333333" })]
  });
}

// Normal paragraph
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Malgun Gothic", size: opts.size || 19, bold: !!opts.bold, color: opts.color || "000000", italics: !!opts.italic })]
  });
}

// Table width constants (A4, 25mm margins)
const TW = 9026; // total content width

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u25CF", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 500, hanging: 250 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u2013", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 900, hanging: 250 } } } },
        ]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Malgun Gothic", size: 19 } }
    }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1200, right: 1440, bottom: 1200, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "NC-SSM Smart Home Voice Control  |  ", font: "Malgun Gothic", size: 14, color: "888888" }),
                     new TextRun({ text: "Confidential", font: "Malgun Gothic", size: 14, color: "CC0000", italics: true })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "- ", font: "Malgun Gothic", size: 16, color: "888888" }),
                     new TextRun({ children: [PageNumber.CURRENT], font: "Malgun Gothic", size: 16, color: "888888" }),
                     new TextRun({ text: " -", font: "Malgun Gothic", size: 16, color: "888888" })]
        })]
      })
    },
    children: [
      // ===== TITLE =====
      new Paragraph({ spacing: { before: 600, after: 100 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "\uCD08\uACA9\uCC28 \uC2A4\uD0C0\uD2B8\uC5C5 \uD504\uB85C\uC81D\uD2B8", font: "Malgun Gothic", size: 22, color: "666666" })] }),
      new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "\uBAA8\uB450\uC758 \uCC4C\uB9B0\uC9C0 AX \u2013 \uBC84\uD2F0\uCEEC \uBD84\uC57C \uD611\uC5C5\uACFC\uC81C \uC218\uD589\uACC4\uD68D\uC11C", font: "Malgun Gothic", size: 32, bold: true, color: "1B3A5C" })] }),
      new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.CENTER,
        border: { bottom: { style: BorderStyle.SINGLE, size: 3, color: "1B3A5C", space: 8 } },
        children: [] }),

      // Info box
      new Table({
        width: { size: TW, type: WidthType.DXA },
        columnWidths: [2000, 7026],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "E8F0FE", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "\uC2E0\uCCAD\uACFC\uC81C", bold: true, font: "Malgun Gothic", size: 18 })] })] }),
            new TableCell({ borders, width: { size: 7026, type: WidthType.DXA }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "LG\uC804\uC790 HS\uBCF8\uBD80 > \uC2A4\uB9C8\uD2B8 \uAC00\uC804 AI \uC194\uB8E8\uC158 > \uC138\uD0C1\uAE30/\uAC74\uC870\uAE30", font: "Malgun Gothic", size: 18 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "E8F0FE", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "\uD611\uC5C5 \uC138\uBD80 \uACFC\uC81C", bold: true, font: "Malgun Gothic", size: 18 })] })] }),
            new TableCell({ borders, width: { size: 7026, type: WidthType.DXA }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "\uCD08\uACBD\uB7C9 \uC628\uB514\uBC14\uC774\uC2A4 AI \uC74C\uC131 \uC778\uC2DD \uAE30\uBC18 \uC81C\uC5B4 \uBC0F \uC784\uBCA0\uB514\uB4DC \uC2DC\uC2A4\uD15C \uD1B5\uD569 \uAE30\uC220\uAC1C\uBC1C", font: "Malgun Gothic", size: 18 })] })] }),
          ]}),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "E8F0FE", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "\uC2E0\uCCAD\uC720\uD615", bold: true, font: "Malgun Gothic", size: 18 })] })] }),
            new TableCell({ borders, width: { size: 7026, type: WidthType.DXA }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "\uD558\uC7041 + \uD558\uC7042 \uD1B5\uD569", font: "Malgun Gothic", size: 18, bold: true, color: "CC0000" })] })] }),
          ]}),
        ]
      }),

      // ===== SECTION: 신청과제명 =====
      sectionHeading("\uC2E0\uCCAD\uACFC\uC81C\uBA85"),
      new Paragraph({ spacing: { before: 120, after: 200 }, alignment: AlignmentType.CENTER,
        shading: { fill: "F0F7FF", type: ShadingType.CLEAR },
        children: [new TextRun({ text: "NC-SSM \uAE30\uBC18 \uCD08\uACBD\uB7C9 \uC628\uB514\uBC14\uC774\uC2A4 \uC74C\uC131 \uC778\uC2DD \uBC0F \uBB38\uB9E5 \uAE30\uBC18 \uAC00\uC804 \uC81C\uC5B4 \uD1B5\uD569 \uC194\uB8E8\uC158", font: "Malgun Gothic", size: 24, bold: true, color: "1B3A5C" })] }),

      // ===== 3-1 협업과제 소개 =====
      sectionHeading("3-1. \uD611\uC5C5\uACFC\uC81C \uC18C\uAC1C"),
      subHeading("\uC81C\uC548 \uBC30\uACBD"),
      bullet("\uD604\uC7AC \uC2A4\uB9C8\uD2B8 \uAC00\uC804\uC758 \uC74C\uC131 \uC778\uC2DD\uC740 \uB300\uBD80\uBD84 \uD074\uB77C\uC6B0\uB4DC \uC11C\uBC84\uC5D0 \uC758\uC874\uD558\uC5EC \uC751\uB2F5 \uC9C0\uC5F0(200ms~1s), \uB124\uD2B8\uC6CC\uD06C \uC758\uC874\uC131, \uAC1C\uC778\uC815\uBCF4 \uC720\uCD9C \uC704\uD5D8, \uC9C0\uC18D\uC801 \uD1B5\uC2E0 \uBE44\uC6A9\uC774 \uBC1C\uC0DD\uD568"),
      subBullet("\uC138\uD0C1\uAE30/\uAC74\uC870\uAE30\uB294 \uB124\uD2B8\uC6CC\uD06C \uBD88\uC548\uC815 \uD658\uACBD(\uC9C0\uD558\uC2E4, \uC138\uD0C1\uC2E4)\uC5D0\uC11C \uC0AC\uC6A9\uB418\uB294 \uACBD\uC6B0\uAC00 \uB9CE\uC544 \uD074\uB77C\uC6B0\uB4DC \uAE30\uBC18 \uC74C\uC131 \uC778\uC2DD\uC758 \uD55C\uACC4\uAC00 \uB450\uB4DC\uB7EC\uC9D0"),
      subBullet("\uC18C\uC74C \uD658\uACBD(\uC138\uD0C1\uAE30 \uD0C8\uC218 \uC9C4\uB3D9 70~80dB, \uAC74\uC870\uAE30 \uC791\uB3D9\uC74C 60~70dB)\uC5D0\uC11C \uAE30\uC874 \uC628\uB514\uBC14\uC774\uC2A4 \uACBD\uB7C9 \uBAA8\uB378\uC758 \uC778\uC2DD\uB960\uC774 \uAE09\uACA9\uD788 \uD558\uB77D(96% \u2192 28% @-15dB)"),
      bullet("\uBCF8 \uACFC\uC81C\uB294 \uC790\uCCB4 \uAC1C\uBC1C\uD55C NC-SSM(Noise-Conditioned State Space Model) \uAE30\uC220\uC744 \uAE30\uBC18\uC73C\uB85C, 4.8KB \uCD08\uACBD\uB7C9 \uBAA8\uB378\uB85C \uC18C\uC74C \uD658\uACBD\uC5D0\uC11C\uB3C4 95%+ \uC778\uC2DD\uB960\uC744 \uB2EC\uC131\uD558\uB294 \uC644\uC804 \uC628\uB514\uBC14\uC774\uC2A4 \uC74C\uC131 \uC778\uC2DD \uBC0F \uAC00\uC804 \uC81C\uC5B4 \uD1B5\uD569 \uC194\uB8E8\uC158\uC744 \uAC1C\uBC1C\uD568"),
      subBullet("\uD558\uC7041(NLU \uAC00\uC804 \uC81C\uC5B4)\uACFC \uD558\uC7042(\uCD08\uACBD\uB7C9 \uC784\uBCA0\uB514\uB4DC \uCD5C\uC801\uD654)\uB97C \uB2E8\uC77C \uD30C\uC774\uD504\uB77C\uC778\uC73C\uB85C \uD1B5\uD569"),
      subBullet("\uAE30\uC874 AI Glass \uD504\uB85C\uC81D\uD2B8\uC5D0\uC11C \uAC80\uC99D\uB41C KWS\u2192STT\u2192NLU\u2192\uC81C\uC5B4 \uD30C\uC774\uD504\uB77C\uC778\uC744 LG \uAC00\uC804 \uD658\uACBD\uC5D0 \uB9DE\uCDA4 \uCD5C\uC801\uD654"),

      subHeading("12-Class KWS \uAE30\uBC18 \uAC00\uC804 \uC81C\uC5B4 \uB9E4\uD551"),
      para("NC-SSM\uC774 \uC778\uC2DD\uD558\uB294 12\uAC1C \uD0A4\uC6CC\uB4DC(Google Speech Commands V2)\uB97C \uAC00\uC804 \uC81C\uC5B4 \uBA85\uB839\uC73C\uB85C \uC9C1\uC811 \uB9E4\uD551\uD558\uC5EC, \uBCC4\uB3C4\uC758 NLU \uC5C6\uC774\uB3C4 MCU \uB2E8\uB3C5\uC73C\uB85C \uAC00\uC804 \uC81C\uC5B4\uAC00 \uAC00\uB2A5\uD568:"),

      // KWS mapping table
      new Table({
        width: { size: TW, type: WidthType.DXA },
        columnWidths: [1800, 2000, 5226],
        rows: [
          new TableRow({ children: [hCell("\uD0A4\uC6CC\uB4DC", 1800), hCell("\uAE30\uB2A5", 2000), hCell("\uB3D9\uC791 \uC608\uC2DC", 5226)] }),
          new TableRow({ children: [bCell("LEFT / RIGHT", 1800, { center: true, bold: true, color: "6366F1" }), bCell("\uAE30\uAE30 \uC804\uD658", 2000), bCell("\uC138\uD0C1\uAE30 \u2192 \uAC74\uC870\uAE30 \u2192 \uC5D0\uC5B4\uCF58 \u2192 \uC870\uBA85 \u2192 TV \uC21C\uD658", 5226)] }),
          new TableRow({ children: [bCell("ON / OFF", 1800, { center: true, bold: true, color: "06B6D4" }), bCell("\uC804\uC6D0 \uC81C\uC5B4", 2000), bCell("\uC120\uD0DD\uB41C \uAE30\uAE30 \uC804\uC6D0 \uCF1C\uAE30/\uB044\uAE30", 5226)] }),
          new TableRow({ children: [bCell("GO", 1800, { center: true, bold: true, color: "06B6D4" }), bCell("\uB3D9\uC791 \uC2DC\uC791/\uC7AC\uAC1C", 2000), bCell("\uC138\uD0C1 \uC2DC\uC791, \uAC74\uC870 \uC2DC\uC791", 5226)] }),
          new TableRow({ children: [bCell("STOP", 1800, { center: true, bold: true, color: "06B6D4" }), bCell("\uC77C\uC2DC\uC815\uC9C0", 2000), bCell("\uB3D9\uC791 \uC911 \uC815\uC9C0", 5226)] }),
          new TableRow({ children: [bCell("UP / DOWN", 1800, { center: true, bold: true, color: "22C55E" }), bCell("\uD30C\uB77C\uBBF8\uD130 \uC870\uC815", 2000), bCell("\uC138\uD0C1\uAE30: \uC628\uB3C4 \u00B110\u00B0C / \uC5D0\uC5B4\uCF58: \u00B11\u00B0C / \uC870\uBA85: \uBC1D\uAE30 \u00B120% / TV: \uCC44\uB110 \u00B11", 5226)] }),
          new TableRow({ children: [bCell("YES", 1800, { center: true, bold: true, color: "F59E0B" }), bCell("\uBAA8\uB4DC \uC21C\uD658", 2000), bCell("\uD45C\uC900\u2192\uC6B8\u2192\uC774\uBD88\u2192\uAE09\uC18D\u2192\uD5F9\uAD74\u2192\uD0C8\uC218\u2192\uC5D0\uCF54 \uC21C\uD658", 5226)] }),
          new TableRow({ children: [bCell("NO", 1800, { center: true, bold: true, color: "F59E0B" }), bCell("\uC124\uC815 \uCD08\uAE30\uD654", 2000), bCell("\uBAA8\uB4E0 \uD30C\uB77C\uBBF8\uD130\uB97C \uAE30\uBCF8\uAC12\uC73C\uB85C \uB9AC\uC14B", 5226)] }),
        ]
      }),
      para("\u2192 KWS \uBAA8\uB378(7,443 params, 4.8KB) \uD558\uB098\uB9CC\uC73C\uB85C \uC138\uD0C1\uAE30/\uAC74\uC870\uAE30\uC758 \uC804\uC6D0, \uBAA8\uB4DC, \uC628\uB3C4, \uB3D9\uC791\uC744 \uBAA8\uB450 \uC81C\uC5B4 \uAC00\uB2A5. MCU \uB2E8\uB3C5 \uB3D9\uC791\uC73C\uB85C \uCD94\uAC00 NLU \uBAA8\uB4C8\uC774\uB098 \uD074\uB77C\uC6B0\uB4DC\uAC00 \uC804\uD600 \uD544\uC694 \uC5C6\uC74C.", { italic: true, color: "2E75B6", size: 18 }),

      subHeading("\uC8FC\uC694 \uAE30\uB2A5 \uBC0F \uC131\uB2A5"),
      // Performance table
      new Table({
        width: { size: TW, type: WidthType.DXA },
        columnWidths: [2500, 6526],
        rows: [
          new TableRow({ children: [hCell("\uD56D\uBAA9", 2500), hCell("\uC0AC\uC591", 6526)] }),
          new TableRow({ children: [bCell("\uBAA8\uB378 \uD06C\uAE30", 2500, { shade: "F5F5F5" }), bCell("7,443 \uD30C\uB77C\uBBF8\uD130 / INT8 4.8KB", 6526)] }),
          new TableRow({ children: [bCell("\uCD94\uB860 \uC9C0\uC5F0\uC2DC\uAC04", 2500, { shade: "F5F5F5" }), bCell("0.94ms (Cortex-M7 480MHz)", 6526)] }),
          new TableRow({ children: [bCell("\uBA54\uBAA8\uB9AC \uC0AC\uC6A9", 2500, { shade: "F5F5F5" }), bCell("RAM 23.8KB, Flash 7.3KB", 6526)] }),
          new TableRow({ children: [bCell("\uBC30\uD130\uB9AC \uC218\uBA85", 2500, { shade: "F5F5F5" }), bCell("CR2032 \uCF54\uC778\uC140 288\uC77C \uC5F0\uC18D \uB3D9\uC791", 6526)] }),
          new TableRow({ children: [bCell("\uC778\uC2DD \uC815\uD655\uB3C4", 2500, { shade: "F5F5F5" }), bCell("95.1% (\uD074\uB9B0) / 71.3% (-15dB \uB178\uC774\uC988)", 6526)] }),
          new TableRow({ children: [bCell("\uB178\uC774\uC988 \uAC15\uAC74\uC131", 2500, { shade: "F5F5F5" }), bCell("\uBC31\uC0C9\uC18C\uC74C -15dB\uC5D0\uC11C +23.8%p \uAC1C\uC120 (vs CNN \uB300\uBE44)", 6526)] }),
          new TableRow({ children: [bCell("\uC9C0\uC6D0 MCU", 2500, { shade: "F5F5F5" }), bCell("ARM Cortex-M4/M7/M33/M55, STM32, NXP i.MX RT", 6526)] }),
          new TableRow({ children: [bCell("\uC5F0\uC0B0\uD6A8\uC728", 2500, { shade: "F5F5F5" }), bCell("0.68M MACs (BC-ResNet-1 \uB300\uBE44 6.8\uBC30 \uD6A8\uC728)", 6526)] }),
          new TableRow({ children: [bCell("\uC81C\uC5B4 \uBC94\uC704", 2500, { shade: "F5F5F5" }), bCell("12-class KWS\uB85C \uC804\uC6D0/\uBAA8\uB4DC/\uC628\uB3C4/\uB3D9\uC791 \uC644\uC804 \uC81C\uC5B4", 6526)] }),
        ]
      }),

      // ===== 3-2 기술 경쟁력 =====
      sectionHeading("3-2. \uD611\uC5C5\uACFC\uC81C \uAE30\uC220 \uACBD\uC7C1\uB825"),
      subHeading("\uD575\uC2EC \uAE30\uC220 \uD601\uC2E0\uC131: SSM \uB178\uC774\uC988 \uC804\uD30C \uBB38\uC81C\uC758 \uC138\uACC4 \uCD5C\uCD08 \uC218\uD559\uC801 \uD574\uACB0"),
      bullet("\uAE30\uC874 \uC120\uD0DD\uC801 SSM(Mamba)\uC740 \uC785\uB825 \uC758\uC874 \uD30C\uB77C\uBBF8\uD130(\u0394, B, C)\uAC00 \uB178\uC774\uC988\uC640 \uACF1\uC148\uC801\uC73C\uB85C \uACB0\uD569\uD558\uC5EC \uCD9C\uB825 \uBD84\uC0B0\uC774 O(\u03C3_n^6)\uC73C\uB85C \uD3ED\uC99D"),
      bullet("NC-SSM\uC740 per-sub-band \uC120\uD0DD\uC131 \uAC8C\uC774\uD305\uC73C\uB85C 40\uAC1C \uBA5C \uBC34\uB4DC\uB97C 6\uAC1C \uC11C\uBE0C\uBC34\uB4DC\uB85C \uADF8\uB8F9\uD654, \uCD9C\uB825 \uBD84\uC0B0\uC744 CNN\uAE09 O(\u03C3_n^2)\uC73C\uB85C \uC81C\uC5B4"),
      subHeading("\uACBD\uC7C1 \uAE30\uC220 \uB300\uBE44 \uCC28\uBCC4\uC131"),
      bullet("vs Qualcomm BC-ResNet-1 (7,464 params): \uB3D9\uC77C \uD30C\uB77C\uBBF8\uD130 \uC218\uC5D0\uC11C \uC5F0\uC0B0\uB7C9 6.8\uBC30 \uC808\uAC10, \uBC30\uD130\uB9AC \uC218\uBA85 6.8\uBC30 \uC5F0\uC7A5, \uB178\uC774\uC988 -15dB\uC5D0\uC11C +23.8%p \uC815\uD655\uB3C4 \uC6B0\uC704"),
      bullet("vs Google DS-CNN-S (23,756 params): 3.2\uBC30 \uC801\uC740 \uD30C\uB77C\uBBF8\uD130\uB85C \uB3D9\uB4F1 \uC774\uC0C1 \uC131\uB2A5"),
      bullet("vs Picovoice/Sensory \uC0C1\uC6A9 \uC194\uB8E8\uC158: 10~100\uBC30 \uB354 \uC791\uC740 \uBAA8\uB378 \uD06C\uAE30\uB85C \uB3D9\uB4F1 \uB178\uC774\uC988 \uAC15\uAC74\uC131"),
      subHeading("\uAE30\uC220 \uC644\uC131 \uC218\uC900 (TRL 4~5)"),
      bullet("\uD559\uC220 \uAC80\uC99D: Interspeech 2026 \uAD6D\uC81C\uD559\uC220\uB300\uD68C \uB17C\uBB38 \uD22C\uACE0 \uC644\uB8CC"),
      bullet("\uC9C0\uC801\uC7AC\uC0B0\uAD8C: \uD55C\uAD6D \uD2B9\uD5C8 \uCD9C\uC6D0 \uC644\uB8CC (NC-SSM + DualPCEN + MoE \uB77C\uC6B0\uD305), \uBBF8\uAD6D \uD2B9\uD5C8 \uCD9C\uC6D0 \uC900\uBE44 \uC911"),
      bullet("\uC18C\uD504\uD2B8\uC6E8\uC5B4: C SDK, Python SDK, ONNX/TFLite \uB0B4\uBCF4\uB0B4\uAE30, \uC2A4\uD2B8\uB9AC\uBC0D \uCD94\uB860 \uC5D4\uC9C4 \uAD6C\uD604 \uC644\uB8CC"),
      bullet("\uC2E4\uC2DC\uAC04 \uB370\uBAA8: CES 2027 \uB77C\uC774\uBE0C \uB370\uBAA8, AI Glass \uBA40\uD2F0\uBAA8\uB2EC \uB370\uBAA8 \uC2DC\uC2A4\uD15C \uC6B4\uC601 \uACBD\uD5D8"),

      // ===== 4-1 추진전략 =====
      sectionHeading("4-1. \uD611\uC5C5 \uBAA9\uD45C \uBC0F \uCD94\uC9C4\uC804\uB7B5"),
      bullet("\uCD5C\uC885 \uBAA9\uD45C: LG \uC138\uD0C1\uAE30/\uAC74\uC870\uAE30 \uC784\uBCA0\uB514\uB4DC MCU\uC5D0 NC-SSM \uAE30\uBC18 \uC628\uB514\uBC14\uC774\uC2A4 \uC74C\uC131 \uC778\uC2DD + 12-Class KWS \uAC00\uC804 \uC81C\uC5B4 \uBAA8\uB4C8 \uD0D1\uC7AC"),
      subBullet("\uD558\uC7042: NC-SSM \uBAA8\uB378\uC744 LG \uAC00\uC804 \uD0C0\uAC9F MCU(STM32/NXP)\uC5D0 \uCD5C\uC801 \uD3EC\uD305, INT8 \uC591\uC790\uD654, 24KB RAM \uB0B4 \uB3D9\uC791 \uAC80\uC99D"),
      subBullet("\uD558\uC7041: 12-Class KWS \uAE30\uBC18 \uAC00\uC804 \uC81C\uC5B4 \uC778\uD130\uD398\uC774\uC2A4 + \uD55C\uAD6D\uC5B4 NLU \uC5D4\uC9C4 \uD1B5\uD569"),
      bullet("\uCD94\uC9C4\uC804\uB7B5: 3\uB2E8\uACC4 \uC810\uC9C4\uC801 \uD1B5\uD569"),
      subBullet("1\uB2E8\uACC4 (1~4\uAC1C\uC6D4): \uBAA8\uB378 \uCD5C\uC801\uD654 \uBC0F \uD558\uB4DC\uC6E8\uC5B4 \uD3EC\uD305 + 12-Class KWS \uAC00\uC804 \uC81C\uC5B4 \uC778\uD130\uD398\uC774\uC2A4 \uAD6C\uD604"),
      subBullet("2\uB2E8\uACC4 (5~8\uAC1C\uC6D4): NLU \uD1B5\uD569 \uBC0F \uAC00\uC804 \uC81C\uC5B4 API \uC5F0\uB3D9"),
      subBullet("3\uB2E8\uACC4 (9~12\uAC1C\uC6D4): PoC \uAC80\uC99D \uBC0F \uCD5C\uC801\uD654"),

      // ===== 4-3 수행계획 =====
      sectionHeading("4-3. \uD611\uC5C5\uACFC\uC81C \uC218\uD589\uACC4\uD68D"),
      subHeading("1\uB2E8\uACC4: \uBAA8\uB378 \uCD5C\uC801\uD654 \uBC0F \uD558\uB4DC\uC6E8\uC5B4 \uD3EC\uD305 (1~4\uAC1C\uC6D4)"),
      bullet("LG \uC138\uD0C1\uAE30/\uAC74\uC870\uAE30 \uD0C0\uAC9F MCU \uD655\uC815 \uBC0F \uAC1C\uBC1C\uD658\uACBD \uAD6C\uCD95"),
      bullet("NC-SSM \uBAA8\uB378 INT8 \uC591\uC790\uD654 \u2192 C SDK \uD3EC\uD305 \u2192 \uD0C0\uAC9F MCU \uB3D9\uC791 \uAC80\uC99D"),
      bullet("12-Class KWS \uAC00\uC804 \uC81C\uC5B4 \uC778\uD130\uD398\uC774\uC2A4 \uAD6C\uD604: 10\uAC1C \uD0A4\uC6CC\uB4DC \u2192 \uC804\uC6D0/\uBAA8\uB4DC/\uC628\uB3C4/\uB3D9\uC791 \uC81C\uC5B4 \uB9E4\uD551"),
      bullet("\uD55C\uAD6D\uC5B4 \uCEE4\uC2A4\uD140 \uD0A4\uC6CC\uB4DC \uD559\uC2B5 \uB370\uC774\uD130\uC14B \uC218\uC9D1 (\"\uCF1C\uC918\", \"\uAEBC\uC918\", \"\uC2DC\uC791\", \"\uBA48\uCDB0\" \uB4F1)"),
      bullet("\uC138\uD0C1\uAE30/\uAC74\uC870\uAE30 \uC791\uB3D9 \uC18C\uC74C \uD504\uB85C\uD30C\uC77C \uC218\uC9D1 \uBC0F \uB178\uC774\uC988 \uAC15\uAC74\uC131 \uCD5C\uC801\uD654"),
      subHeading("2\uB2E8\uACC4: NLU \uD1B5\uD569 \uBC0F \uAC00\uC804 \uC81C\uC5B4 (5~8\uAC1C\uC6D4)"),
      bullet("\uD55C\uAD6D\uC5B4 \uBB38\uB9E5 \uC778\uC2DD NLU \uC5D4\uC9C4 \uAC1C\uBC1C (\uC758\uB3C4/\uC2AC\uB86F \uBD84\uB958: \uAE30\uAE30, \uB3D9\uC791, \uD30C\uB77C\uBBF8\uD130)"),
      bullet("LG \uAC00\uC804 \uC81C\uC5B4 API \uC5F0\uB3D9 \uBBF8\uB4E4\uC6E8\uC5B4 \uAC1C\uBC1C"),
      bullet("\uBCF5\uD569 \uBA85\uB839 \uCC98\uB9AC: \"\uC138\uD0C1 \uB05D\uB098\uBA74 \uAC74\uC870\uAE30 \uC800\uC628\uC73C\uB85C 60\uBD84 \uB3CC\uB824\uC918\""),
      subHeading("3\uB2E8\uACC4: PoC \uAC80\uC99D \uBC0F \uCD5C\uC801\uD654 (9~12\uAC1C\uC6D4)"),
      bullet("LG \uC2E4\uC81C \uC138\uD0C1\uAE30/\uAC74\uC870\uAE30 \uD558\uB4DC\uC6E8\uC5B4\uC5D0 \uBAA8\uB4C8 \uD0D1\uC7AC PoC"),
      bullet("\uC2E4\uC0AC\uC6A9 \uD658\uACBD \uD14C\uC2A4\uD2B8: \uC138\uD0C1\uC2E4 \uC18C\uC74C(70~80dB), \uB2E4\uC591\uD55C \uAC70\uB9AC(1~5m), \uB2E4\uD654\uC790"),
      bullet("\uAE30\uC220\uC774\uC804 \uBB38\uC11C\uD654: API \uBB38\uC11C, \uD1B5\uD569 \uAC00\uC774\uB4DC, \uC720\uC9C0\uBCF4\uC218 \uB9E4\uB274\uC5BC"),

      // ===== 5-1 사업성 =====
      sectionHeading("5-1. \uD611\uC5C5\uACFC\uC81C \uC0AC\uC5C5\uC131"),
      bullet("\uBAA9\uD45C \uC2DC\uC7A5: \uAE00\uB85C\uBC8C \uC5E3\uC9C0 AI \uC74C\uC131 \uC778\uC2DD \uC2DC\uC7A5\uC740 2025\uB144 $15B\uC5D0\uC11C 2030\uB144 $50B+\uB85C \uC5F0 25%+ \uC131\uC7A5 \uC804\uB9DD"),
      bullet("LG\uC804\uC790 \uC601\uD5A5: \uD074\uB77C\uC6B0\uB4DC \uC5C6\uC774 \uC644\uC804\uD55C \uC74C\uC131 \uC81C\uC5B4, BOM \uBE44\uC6A9 \uC808\uAC10, \uC804 \uAC00\uC804 \uB77C\uC778\uC5C5 \uD655\uC7A5 \uAC00\uB2A5"),
      bullet("IP \uB77C\uC774\uC120\uC2F1: $0.01~0.05/unit \uB85C\uC5F4\uD2F0 (LG \uAC00\uC804 \uCD9C\uD558\uB7C9 \uAE30\uC900 \uC5F0 \uC218\uC2ED\uC5B5 \uC6D0 \uADDC\uBAA8)"),

      // ===== 5-2 확장성 =====
      sectionHeading("5-2. \uD655\uC7A5\u00B7\uC9C0\uC18D \uAC00\uB2A5\uC131"),
      bullet("\uC218\uD3C9 \uD655\uC7A5: LG \uAC00\uC804 \uC804 \uB77C\uC778\uC5C5 (\uB0C9\uC7A5\uACE0, \uC5D0\uC5B4\uCF58, \uACF5\uAE30\uCCAD\uC815\uAE30, \uB85C\uBD07\uCCAD\uC18C\uAE30)"),
      bullet("\uC218\uC9C1 \uD655\uC7A5: ARM Cortex-M MCU\uC5D0 NC-SSM RTL/Verilog IP \uD0D1\uC7AC (\uD0C0\uAC9F 31\uAC1C\uC0AC)"),
      bullet("\uC0B0\uC5C5 \uD655\uC7A5: \uC790\uB3D9\uCC28(\uD604\uB300\uBAA8\uBE44\uC2A4, Continental), \uC0B0\uC5C5 IoT(Siemens, ABB), \uC6E8\uC5B4\uB7EC\uBE14(Apple, Sony)"),

      // ===== 7. 사업비 =====
      sectionHeading("7. \uC0AC\uC5C5\uBE44 \uC9D1\uD589\uACC4\uD68D"),
      new Table({
        width: { size: TW, type: WidthType.DXA },
        columnWidths: [1800, 5226, 2000],
        rows: [
          new TableRow({ children: [hCell("\uD56D\uBAA9", 1800), hCell("\uC0B0\uCD9C\uADFC\uAC70", 5226), hCell("\uC0AC\uC6A9\uAE08\uC561(\uC6D0)", 2000)] }),
          new TableRow({ children: [bCell("\uC778\uAC74\uBE44", 1800, { shade: "F5F5F5" }), bCell("AI \uBAA8\uB378 \uAC1C\uBC1C\uC790 2\uBA85 \u00D7 12\uAC1C\uC6D4 + NLU/SW \uAC1C\uBC1C\uC790 1\uBA85 \u00D7 12\uAC1C\uC6D4", 5226), bCell("50,000,000", 2000, { center: true })] }),
          new TableRow({ children: [bCell("\uAE30\uC790\uC7AC \uAD6C\uC785\uBE44", 1800, { shade: "F5F5F5" }), bCell("MCU \uBCF4\uB4DC(STM32H7, NXP i.MX RT) 5\uC885 + \uB9C8\uC774\uD06C \uBAA8\uB4C8 + \uD14C\uC2A4\uD2B8 \uC7A5\uBE44", 5226), bCell("15,000,000", 2000, { center: true })] }),
          new TableRow({ children: [bCell("SW \uAD6C\uC785\uBE44", 1800, { shade: "F5F5F5" }), bCell("ARM Keil MDK \uB77C\uC774\uC120\uC2A4, \uC624\uB514\uC624 \uB370\uC774\uD130\uC14B \uB77C\uC774\uC120\uC2F1", 5226), bCell("10,000,000", 2000, { center: true })] }),
          new TableRow({ children: [bCell("\uC2DC\uD5D8\u00B7\uC778\uC99D\uBE44", 1800, { shade: "F5F5F5" }), bCell("\uC74C\uC131\uC778\uC2DD \uC815\uD655\uB3C4 \uC81C3\uC790 \uAC80\uC99D, EMC/\uC548\uC804 \uC778\uC99D \uD14C\uC2A4\uD2B8", 5226), bCell("10,000,000", 2000, { center: true })] }),
          new TableRow({ children: [bCell("\uC678\uC8FC\uC6A9\uC5ED\uBE44", 1800, { shade: "F5F5F5" }), bCell("\uD55C\uAD6D\uC5B4 \uC74C\uC131 \uB370\uC774\uD130 \uC218\uC9D1/\uB77C\uBCA8\uB9C1 (1\uB9CC \uBC1C\uD654), PCB \uC2DC\uC81C\uD488", 5226), bCell("10,000,000", 2000, { center: true })] }),
          new TableRow({ children: [bCell("\uC5EC\uBE44", 1800, { shade: "F5F5F5" }), bCell("LG\uC804\uC790 \uD604\uC7A5 \uBC29\uBB38 (PoC \uC124\uCE58/\uD14C\uC2A4\uD2B8), \uAE30\uC220 \uBBF8\uD305", 5226), bCell("5,000,000", 2000, { center: true })] }),
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 1800, type: WidthType.DXA }, shading: { fill: "1B3A5C", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\uD569\uACC4", bold: true, color: "FFFFFF", font: "Malgun Gothic", size: 18 })] })] }),
            new TableCell({ borders, width: { size: 5226, type: WidthType.DXA }, margins: cellMargins, children: [new Paragraph({ children: [] })] }),
            new TableCell({ borders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "E8F0FE", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "100,000,000", bold: true, font: "Malgun Gothic", size: 20, color: "CC0000" })] })] }),
          ] }),
        ]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("C:/Users/jinho/Downloads/NanoMamba-Interspeech2026/paper/proposal_ax_challenge.docx", buffer);
  console.log("DOCX created successfully!");
});
