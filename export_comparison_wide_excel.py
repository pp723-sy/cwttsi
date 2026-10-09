"""Export baseline comparison results as a publication-style wide Excel table."""

from __future__ import annotations

import argparse
import csv
import html
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DATASETS = ["XMBRT", "HZMetro", "BJMetro"]
MODEL_HEADER = "\u6a21\u578b"
SHEET_NAME = "\u5bf9\u6bd4\u5b9e\u9a8c\u5bbd\u8868"
MODELS = [
    "LSTM",
    "ASTGCN-r",
    "GraphWaveNet",
    "GMAN",
    "STSGCN",
    "StemGNN",
    "STFGNN",
    "ASTGNN",
    "CWT-TSI",
]
METRICS = [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE(%)")]


def col_name(n: int) -> str:
    result = ""
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def read_comparison(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values[(row["dataset"], row["model"])] = {
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "mape": float(row["mape"]),
            }
    return values


def build_rows(values: dict[tuple[str, str], dict[str, float]]) -> list[list[str | float]]:
    rows: list[list[str | float]] = []
    rows.append([MODEL_HEADER, "XMBRT", "", "", "HZMetro", "", "", "BJMetro", "", ""])
    rows.append(["", "MAE", "RMSE", "MAPE(%)", "MAE", "RMSE", "MAPE(%)", "MAE", "RMSE", "MAPE(%)"])
    for model in MODELS:
        row: list[str | float] = [model]
        for dataset in DATASETS:
            for metric, _ in METRICS:
                row.append(values[(dataset, model)][metric])
        rows.append(row)
    return rows


def best_cells(values: dict[tuple[str, str], dict[str, float]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dataset_idx, dataset in enumerate(DATASETS):
        for metric_idx, (metric, _) in enumerate(METRICS):
            best_model = min(MODELS, key=lambda model: values[(dataset, model)][metric])
            row = 3 + MODELS.index(best_model)
            col = 2 + dataset_idx * 3 + metric_idx
            cells.add((row, col))
    return cells


def style_for_cell(row_idx: int, col_idx: int, best: set[tuple[int, int]]) -> int:
    is_cwt_row = row_idx == 3 + MODELS.index("CWT-TSI")
    is_number = row_idx >= 3 and col_idx >= 2
    if row_idx in (1, 2):
        return 1
    if is_cwt_row and is_number and (row_idx, col_idx) in best:
        return 6
    if is_cwt_row and is_number:
        return 4
    if is_cwt_row:
        return 3
    if is_number and (row_idx, col_idx) in best:
        return 5
    if is_number:
        return 2
    return 0


def sheet_xml(rows: list[list[str | float]], best: set[tuple[int, int]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="2" topLeftCell="A3" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        '<cols><col min="1" max="1" width="18" customWidth="1"/>'
        '<col min="2" max="10" width="12.5" customWidth="1"/></cols>',
        "<sheetData>",
    ]

    for row_idx, row in enumerate(rows, start=1):
        height = ' ht="22" customHeight="1"' if row_idx <= 2 else ""
        parts.append(f'<row r="{row_idx}"{height}>')
        for col_idx, value in enumerate(row, start=1):
            ref = f"{col_name(col_idx)}{row_idx}"
            style = style_for_cell(row_idx, col_idx, best)
            if isinstance(value, float):
                parts.append(f'<c r="{ref}" s="{style}"><v>{value:.15g}</v></c>')
            else:
                parts.append(
                    f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>'
                    f"{html.escape(str(value))}</t></is></c>"
                )
        parts.append("</row>")

    parts.append("</sheetData>")
    parts.append('<mergeCells count="4"><mergeCell ref="A1:A2"/><mergeCell ref="B1:D1"/>'
                 '<mergeCell ref="E1:G1"/><mergeCell ref="H1:J1"/></mergeCells>')
    parts.append(f'<autoFilter ref="A2:J{len(rows)}"/>')
    parts.append("</worksheet>")
    return "".join(parts)


def workbook_parts(sheet: str, now: str) -> dict[str, str]:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>"""
    escaped_sheet_name = html.escape(SHEET_NAME)
    workbook = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{escaped_sheet_name}" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts><fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/><color rgb="FF000000"/></font></fonts><fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FF7FCBE6"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD0D7DE"/></left><right style="thin"><color rgb="FFD0D7DE"/></right><top style="thin"><color rgb="FFD0D7DE"/></top><bottom style="thin"><color rgb="FFD0D7DE"/></bottom><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="7"><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"><alignment horizontal="right" vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="164" fontId="1" fillId="3" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="right" vertical="center"/></xf><xf numFmtId="164" fontId="1" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyBorder="1"><alignment horizontal="right" vertical="center"/></xf><xf numFmtId="164" fontId="1" fillId="3" borderId="1" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="right" vertical="center"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>CWT-TSI comparison wide table</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>"""
    app = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Excel</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs><TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>{escaped_sheet_name}</vt:lpstr></vt:vector></TitlesOfParts></Properties>"""
    return {
        "[Content_Types].xml": content_types,
        "_rels/.rels": rels,
        "xl/workbook.xml": workbook,
        "xl/_rels/workbook.xml.rels": workbook_rels,
        "xl/styles.xml": styles,
        "xl/worksheets/sheet1.xml": sheet,
        "docProps/core.xml": core,
        "docProps/app.xml": app,
    }


def export_csv(rows: list[list[str | float]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument(
        "--csv",
        default="log/comparison_table4_5_full/comparison_table.csv",
        help="Source long-format comparison CSV",
    )
    parser.add_argument(
        "--out",
        default="log/comparison_table4_5_full/comparison_table_wide.xlsx",
        help="Output .xlsx path relative to root",
    )
    args = parser.parse_args()

    root = Path(args.root)
    values = read_comparison(root / args.csv)
    rows = build_rows(values)
    best = best_cells(values)
    sheet = sheet_xml(rows, best)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    parts = workbook_parts(sheet, now)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as z:
        for name, content in parts.items():
            z.writestr(name, content)

    csv_path = out_path.with_suffix(".csv")
    export_csv(rows, csv_path)
    print(f"Output: {out_path}")
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(rows[0])}")


if __name__ == "__main__":
    main()
