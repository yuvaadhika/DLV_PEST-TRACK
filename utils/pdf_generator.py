"""
================================================================================
PestTrack AI - PDF Inspection Report Generator (ReportLab)
================================================================================
"""

import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from typing import Dict, Any, Optional
import config

def generate_pdf_report(record_data: Dict[str, Any], output_path: str) -> bool:
    """
    Generates a professional DLV Agricultural Inspection Report PDF.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        elements = []

        # Color Constants
        PRIMARY = colors.HexColor("#2E7D32")
        DARK_GREEN = colors.HexColor("#1B5E20")
        LIGHT_BG = colors.HexColor("#E8F5E9")
        TEXT_COLOR = colors.HexColor("#1C2826")

        severity = record_data.get("severity_level", "Low")
        if severity == "High":
            sev_color = colors.HexColor("#C62828")
        elif severity == "Medium":
            sev_color = colors.HexColor("#F57C00")
        else:
            sev_color = colors.HexColor("#2E7D32")

        # Header Title
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=PRIMARY,
            alignment=1, # Center
            fontName="Helvetica-Bold"
        )
        subtitle_style = ParagraphStyle(
            "SubTitleStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#555555"),
            alignment=1
        )

        elements.append(Paragraph("PESTTRACK AI: CROP MONITORING SYSTEM", title_style))
        elements.append(Paragraph("Intelligent Pest Infestation & Leaf Damage Analysis Report", subtitle_style))
        elements.append(Spacer(1, 15))

        # Metadata Table
        meta_data = [
            [
                Paragraph("<b>Report ID:</b> PT-" + str(record_data.get("id", "001")), styles["Normal"]),
                Paragraph(f"<b>Date & Time:</b> {record_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}", styles["Normal"])
            ],
            [
                Paragraph(f"<b>Image:</b> {record_data.get('image_name', 'Field_Sample.jpg')}", styles["Normal"]),
                Paragraph(f"<b>Analysis Type:</b> {record_data.get('analysis_type', 'Full Vision Pipeline').title()}", styles["Normal"])
            ]
        ]
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
            ('BOX', (0,0), (-1,-1), 1, PRIMARY),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#C8E6C9")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 15))

        # Key Findings Summary Table
        pest_count = record_data.get("pest_count", 0)
        species = record_data.get("primary_pest_species", "None")
        confidence = record_data.get("confidence", 0.0)
        damage_pct = record_data.get("damage_percentage", 0.0)

        findings_data = [
            ["Metric Parameter", "Observed AI Finding", "Threshold Benchmark"],
            ["Total Pests Detected (YOLOv8)", f"{pest_count} insects", "Low <= 2 | Med: 3-7 | High >= 8"],
            ["Primary Pest Species (PyTorch CNN)", f"{species} ({confidence:.1f}%)", "Minimum 65% Confidence"],
            ["Visible Leaf Damage Area (PyTorch U-Net)", f"{damage_pct:.2f} %", "Low < 10% | Med: 10-30% | High > 30%"],
            ["Infestation Severity Level", f"{severity.upper()}", "Rule Engine Matrix Evaluated"]
        ]

        findings_table = Table(findings_data, colWidths=[180, 180, 180])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F9FBF9")]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#FFF3E0") if severity == "Medium" else (colors.HexColor("#FFEBEE") if severity == "High" else LIGHT_BG)),
            ('TEXTCOLOR', (1, -1), (1, -1), sev_color),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(findings_table)
        elements.append(Spacer(1, 15))

        # Visual Evidence (Images)
        orig_img_path = record_data.get("original_image_path", "")
        res_img_path = record_data.get("result_image_path", "")

        img_cells = []
        if orig_img_path and os.path.exists(orig_img_path):
            try:
                img_cells.append([
                    Paragraph("<b>Original Sample Image:</b>", styles["Normal"]),
                    RLImage(orig_img_path, width=240, height=160)
                ])
            except Exception:
                pass

        if res_img_path and os.path.exists(res_img_path):
            try:
                img_cells.append([
                    Paragraph("<b>AI Processed Detection & Mask:</b>", styles["Normal"]),
                    RLImage(res_img_path, width=240, height=160)
                ])
            except Exception:
                pass

        if len(img_cells) == 2:
            img_table_data = [
                [img_cells[0][0], img_cells[1][0]],
                [img_cells[0][1], img_cells[1][1]]
            ]
            img_table = Table(img_table_data, colWidths=[270, 270])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            elements.append(KeepTogether([img_table]))
            elements.append(Spacer(1, 15))

        # Action Recommendations
        rec_title = ParagraphStyle("RecTitle", parent=styles["Heading2"], fontSize=12, textColor=DARK_GREEN)
        elements.append(Paragraph("Agronomic Advisory & Action Recommendations:", rec_title))
        
        if severity == "High":
            rec_text = "<b>Urgent Intervention Required:</b> High pest count and extensive leaf damage observed. Implement targeted organic biocontrol (e.g. Bacillus thuringiensis or Neem oil extract) or recommended bio-pesticide immediately. Isolate severely damaged plants."
        elif severity == "Medium":
            rec_text = "<b>Active Monitoring Recommended:</b> Moderate infestation detected. Apply preventive biological repellent spray. Re-inspect the crop field within 48 to 72 hours using PestTrack AI to evaluate pest population progression."
        else:
            rec_text = "<b>Low Severity / Safe Condition:</b> Infestation is minimal and well below economic injury threshold. Continue standard preventive farm hygiene and routine weekly monitoring."

        elements.append(Paragraph(rec_text, styles["Normal"]))
        elements.append(Spacer(1, 15))

        # Scientific Disclaimer (DLV Project Requirement)
        disclaimer_style = ParagraphStyle(
            "DiscStyle",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#777777")
        )
        elements.append(Paragraph(f"<b>Yield Impact Disclaimer:</b> {config.DISCLAIMER_TEXT}", disclaimer_style))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Generated automatically by PestTrack AI Windows Desktop System | Academic DLV Case Study", disclaimer_style))

        # Build Document
        doc.build(elements)
        return True
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return False
