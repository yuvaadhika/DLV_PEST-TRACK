"""
================================================================================
PestTrack AI - Professional Agronomic PDF Inspection Report Generator
Framework: ReportLab
Theme: Agriculture Deep Green & Crisp White Symmetrical Layout
================================================================================
"""

import os
import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from typing import Dict, Any, Optional, Tuple
import config

def _get_proportional_image_dims(img_path: str, max_w: int = 245, max_h: int = 150) -> Tuple[int, int]:
    """Computes aspect-ratio preserved dimensions for ReportLab images."""
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                return (max_w, max_h)
            ratio = min(max_w / w, max_h / h)
            return (max(1, int(w * ratio)), max(1, int(h * ratio)))
    except Exception:
        return (max_w, max_h)

def generate_pdf_report(record_data: Dict[str, Any], output_path: str) -> bool:
    """
    Generates a beautifully aligned, publication-quality Agricultural Inspection Report PDF.
    Total printable width: 540 pt (Letter width 612 pt - 72 pt margins).
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=32,
            bottomMargin=32
        )

        styles = getSampleStyleSheet()
        elements = []

        # Color Palette
        PRIMARY = colors.HexColor("#1B5E20")      # Deep Forest Green
        SECONDARY = colors.HexColor("#2E7D32")    # Primary Green
        ACCENT_BG = colors.HexColor("#E8F5E9")    # Light Mint
        BORDER_COL = colors.HexColor("#C8E6C9")   # Soft Green Border
        TEXT_DARK = colors.HexColor("#1A221E")    # Charcoal
        TEXT_MUTED = colors.HexColor("#555555")

        severity = record_data.get("severity_level", "Low").capitalize()
        if severity == "High":
            sev_color = colors.HexColor("#C62828")
            sev_bg = colors.HexColor("#FFEBEE")
            sev_border = colors.HexColor("#EF9A9A")
            yield_range = "25% - 40%"
        elif severity == "Medium":
            sev_color = colors.HexColor("#E65100")
            sev_bg = colors.HexColor("#FFF3E0")
            sev_border = colors.HexColor("#FFCC80")
            yield_range = "10% - 25%"
        else:
            sev_color = colors.HexColor("#2E7D32")
            sev_bg = colors.HexColor("#E8F5E9")
            sev_border = colors.HexColor("#A5D6A7")
            yield_range = "< 10%"

        # ----------------------------------------------------------------------
        # 1. HEADER BANNER TABLE (Full 540 pt Width)
        # ----------------------------------------------------------------------
        h_title_style = ParagraphStyle(
            "HTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.white,
            alignment=1,
            fontName="Helvetica-Bold"
        )
        h_sub_style = ParagraphStyle(
            "HSub",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#E8F5E9"),
            alignment=1
        )

        header_data = [
            [Paragraph("🌱 PESTTRACK AI : CROP INSPECTION & DAMAGE REPORT", h_title_style)],
            [Paragraph("Deep Learning for Vision (DLV) | Multi-Model Agronomic Advisory System", h_sub_style)]
        ]
        header_table = Table(header_data, colWidths=[540])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PRIMARY),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
            ('TOPPADDING', (0, 1), (-1, 1), 0),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 12))

        # ----------------------------------------------------------------------
        # 2. METADATA SUMMARY TABLE (2 Symmetrical Columns of 270 pt = 540 pt)
        # ----------------------------------------------------------------------
        meta_label_style = ParagraphStyle(
            "MetaLbl",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=TEXT_DARK
        )

        r_id = record_data.get("id", "001")
        t_stamp = record_data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        img_name = str(record_data.get("image_name", "Field_Foliage_Sample.jpg"))[:32]
        analysis_type = str(record_data.get("analysis_type", "Unified DLV Multi-Model Pipeline")).title()

        meta_data = [
            [
                Paragraph(f"<b>Report Reference:</b> PT-REF-{r_id:04d}", meta_label_style),
                Paragraph(f"<b>Inspection Date & Time:</b> {t_stamp}", meta_label_style)
            ],
            [
                Paragraph(f"<b>Sample Image:</b> {img_name}", meta_label_style),
                Paragraph(f"<b>Analysis Scope:</b> {analysis_type}", meta_label_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F9FBF9")),
            ('BOX', (0, 0), (-1, -1), 1, BORDER_COL),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # ----------------------------------------------------------------------
        # 3. KEY DIAGNOSTIC FINDINGS TABLE (160 + 190 + 190 = 540 pt)
        # ----------------------------------------------------------------------
        pest_count = record_data.get("pest_count", 0)
        species = record_data.get("primary_pest_species", "None")
        confidence = record_data.get("confidence", 0.0)
        damage_pct = record_data.get("damage_percentage", 0.0)

        tbl_hdr_style = ParagraphStyle("THdr", parent=styles["Normal"], fontSize=9.5, leading=13, fontName="Helvetica-Bold", textColor=colors.white)
        tbl_cell_style = ParagraphStyle("TCell", parent=styles["Normal"], fontSize=9, leading=13, textColor=TEXT_DARK)
        tbl_bold_style = ParagraphStyle("TBold", parent=styles["Normal"], fontSize=9, leading=13, fontName="Helvetica-Bold", textColor=TEXT_DARK)

        findings_data = [
            [
                Paragraph("<b>Evaluation Parameter</b>", tbl_hdr_style),
                Paragraph("<b>Observed AI Finding</b>", tbl_hdr_style),
                Paragraph("<b>Threshold Benchmark & Model</b>", tbl_hdr_style)
            ],
            [
                Paragraph("<b>Pest Density Count</b>", tbl_cell_style),
                Paragraph(f"<b>{pest_count}</b> detected pests", tbl_bold_style),
                Paragraph("YOLOv8 Object Detection (Low &le;2, Med 3-7, High &ge;8)", tbl_cell_style)
            ],
            [
                Paragraph("<b>Primary Pest Species</b>", tbl_cell_style),
                Paragraph(f"<b>{species}</b> ({confidence:.1%})", tbl_bold_style),
                Paragraph("4-Stage PyTorch CNN (10 Agricultural Classes)", tbl_cell_style)
            ],
            [
                Paragraph("<b>Leaf Damage Area Ratio</b>", tbl_cell_style),
                Paragraph(f"<b>{damage_pct:.2f} %</b> damaged tissue", tbl_bold_style),
                Paragraph("PyTorch U-Net (Low &lt;10%, Med 10-30%, High &gt;30%)", tbl_cell_style)
            ],
            [
                Paragraph("<b>Estimated Yield Loss Range</b>", tbl_cell_style),
                Paragraph(f"<b>{yield_range}</b> potential impact", tbl_bold_style),
                Paragraph("Agronomic Empirical Correlation Matrix", tbl_cell_style)
            ],
            [
                Paragraph("<b>Overall Infestation Severity</b>", tbl_cell_style),
                Paragraph(f"<b>{severity.upper()} SEVERITY</b>", ParagraphStyle("SevTxt", parent=styles["Normal"], fontSize=10, leading=14, fontName="Helvetica-Bold", textColor=sev_color)),
                Paragraph("Rule-Based Multi-Factor Decision Engine", tbl_cell_style)
            ]
        ]

        findings_table = Table(findings_data, colWidths=[160, 190, 190])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F9FBF9")]),
            ('BACKGROUND', (0, -1), (-1, -1), sev_bg),
            ('BOX', (0, -1), (-1, -1), 1, sev_border),
        ]))
        elements.append(findings_table)
        elements.append(Spacer(1, 12))

        # ----------------------------------------------------------------------
        # 4. VISUAL EVIDENCE (Images Grid: 270 + 270 = 540 pt)
        # ----------------------------------------------------------------------
        orig_img_path = record_data.get("original_image_path", "")
        res_img_path = record_data.get("result_image_path", "")

        cap_style = ParagraphStyle(
            "CapStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            fontName="Helvetica-Bold",
            textColor=SECONDARY,
            alignment=1
        )

        img_row_labels = [
            Paragraph("<b>Figure A: Original Crop Foliage Sample</b>", cap_style),
            Paragraph("<b>Figure B: AI YOLO Detection & U-Net Damage Mask</b>", cap_style)
        ]

        cell_a = [Paragraph("<i>Original Image Not Found</i>", styles["Normal"])]
        cell_b = [Paragraph("<i>Processed Image Not Found</i>", styles["Normal"])]

        if orig_img_path and os.path.exists(orig_img_path):
            try:
                iw, ih = _get_proportional_image_dims(orig_img_path, max_w=245, max_h=140)
                cell_a = [RLImage(orig_img_path, width=iw, height=ih)]
            except Exception:
                pass

        if res_img_path and os.path.exists(res_img_path):
            try:
                iw, ih = _get_proportional_image_dims(res_img_path, max_w=245, max_h=140)
                cell_b = [RLImage(res_img_path, width=iw, height=ih)]
            except Exception:
                pass

        img_table_data = [
            img_row_labels,
            [cell_a, cell_b]
        ]
        img_table = Table(img_table_data, colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
            ('BOX', (0, 0), (-1, -1), 1, BORDER_COL),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 1), (-1, 1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 6),
        ]))
        elements.append(KeepTogether([img_table]))
        elements.append(Spacer(1, 12))

        # ----------------------------------------------------------------------
        # 5. AGRONOMIC ADVISORY & TREATMENT RECOMMENDATION (540 pt Width)
        # ----------------------------------------------------------------------
        rec_title_style = ParagraphStyle(
            "RecTitle",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=14,
            fontName="Helvetica-Bold",
            textColor=PRIMARY
        )
        rec_body_style = ParagraphStyle(
            "RecBody",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=12.5,
            textColor=TEXT_DARK
        )

        if severity == "High":
            rec_body = (
                "<b>Immediate Intervention Required:</b> Severe foliar necrosis/chewing and significant pest population observed. "
                "1. Apply targeted biological insecticide (e.g., <i>Bacillus thuringiensis</i> or Azadirachtin / cold-pressed Neem oil extract at 5ml/L). "
                "2. Manually isolate and dispose of heavily infested crop leaves to prevent cross-row migration. "
                "3. Re-survey treated sectors in 48 hours with PestTrack AI to evaluate knockdown efficiency."
            )
        elif severity == "Medium":
            rec_body = (
                "<b>Active Surveillance & Preventive Control:</b> Moderate pest incidence and localized foliage damage recorded. "
                "1. Introduce beneficial predatory biocontrol insects (e.g. Chrysoperla carnea or ladybird beetles) or mild organic repellent spray. "
                "2. Conduct scheduled re-inspection in 48 to 72 hours to ensure pest density remains below economic injury threshold (EIT)."
            )
        else:
            rec_body = (
                "<b>Safe Condition / Routine Hygiene:</b> Pest density and visible leaf tissue damage are minimal and well below economic damage limits. "
                "1. Continue standard preventive irrigation and crop canopy hygiene. "
                "2. Maintain routine weekly vision monitoring."
            )

        advisory_data = [
            [Paragraph("🌿 <b>Agronomic Advisory & Treatment Recommendation:</b>", rec_title_style)],
            [Paragraph(rec_body, rec_body_style)]
        ]
        advisory_table = Table(advisory_data, colWidths=[540])
        advisory_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F9F6")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#81C784")),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
            ('TOPPADDING', (0, 1), (-1, 1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(advisory_table)
        elements.append(Spacer(1, 10))

        # ----------------------------------------------------------------------
        # 6. SCIENTIFIC & ACADEMIC DISCLAIMER FOOTER (540 pt Width)
        # ----------------------------------------------------------------------
        disc_style = ParagraphStyle(
            "DiscStyle",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10.5,
            textColor=TEXT_MUTED,
            alignment=1
        )

        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CFD8DC"), spaceAfter=6))
        elements.append(Paragraph(f"<b>Scientific Impact Notice:</b> {config.DISCLAIMER_TEXT}", disc_style))
        elements.append(Spacer(1, 3))
        elements.append(Paragraph("Automated Document Generated by PestTrack AI System | DLV Vision Case Study | 3rd Year B.Tech AIML", disc_style))

        # Build Document
        doc.build(elements)
        return True
    except Exception as e:
        print(f"Error generating aligned PDF report: {e}")
        return False

def generate_pest_report(output_pdf_path: str, analysis_data: Dict[str, Any], result_image_path: Optional[str] = None) -> bool:
    """Wrapper function ensuring backward & web compatibility."""
    if result_image_path and "result_image_path" not in analysis_data:
        analysis_data["result_image_path"] = result_image_path
    return generate_pdf_report(analysis_data, output_pdf_path)
