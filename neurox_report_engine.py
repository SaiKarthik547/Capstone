import re
import io
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

class StructuredReport:
    """
    Structured Clinical Report Object for NeuroX.
    De-couples raw AI text from the visual presentation layer.
    """
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.findings: List[str] = []
        self.measurements: List[Dict] = []
        self.differential: List[str] = []
        self.impression: str = ""
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self._parse()

    def _parse(self):
        """
        Robust/Heuristic parser for NeuroX Radiological outputs.
        Handles bolding (**), hash headers (##), and numeric/roman headers.
        """
        findings_regex    = r"(?:\*\*|##|#)?\s*(?:[IV\d]+\.?\s*)?(?:CLINICAL\s+)?FINDINGS\s*:?\s*(?:\*\*)?"
        measurements_regex = r"(?:\*\*|##|#)?\s*(?:[IV\d]+\.?\s*)?(?:LOCALIZED\s+)?(?:ANALYTICAL\s+)?MEASUREMENTS\s*:?\s*(?:\*\*)?"
        differential_regex = r"(?:\*\*|##|#)?\s*(?:[IV\d]+\.?\s*)?(?:DIFFERENTIAL\s+)?(?:CONSIDERATIONS|DIAGNOSIS)\s*:?\s*(?:\*\*)?"
        impression_regex   = r"(?:\*\*|##|#)?\s*(?:[IV\d]+\.?\s*)?IMPRESSION\s*:?\s*(?:\*\*)?"

        # 1. Findings
        findings_match = re.search(f"{findings_regex}(.*?)(?={measurements_regex}|{differential_regex}|{impression_regex}|$)", self.raw_text, re.DOTALL | re.IGNORECASE)
        if findings_match:
            lines = findings_match.group(1).strip().split('\n')
            self.findings = [l.strip().lstrip('•').lstrip('+').lstrip('-').lstrip('*').strip() for l in lines if l.strip()]

        # 2. Measurements (Standard Section)
        measurements_match = re.search(f"{measurements_regex}(.*?)(?={differential_regex}|{impression_regex}|$)", self.raw_text, re.DOTALL | re.IGNORECASE)
        if measurements_match:
            lines = measurements_match.group(1).strip().split('\n')
            for line in lines:
                line = line.strip()
                if any(unit in line for unit in ['mL', 'mm', 'RAS', 'Coordinate', 'voxel']):
                    if len(line) < 140 and "|" not in line:
                        m_text = line.lstrip('•').lstrip('+').lstrip('-').lstrip('*').strip()
                        if m_text not in [m["Metric"] for m in self.measurements]:
                            self.measurements.append({"Metric": m_text})

        # 3. SCAVENGE FALLBACK: Trace metrics from the entire body if Measurements is empty or limited
        # ONLY if the Measurements section didn't already capture volumetric data
        has_volume = any('mL' in m['Metric'] for m in self.measurements)
        if not has_volume:
            for line in self.raw_text.split('\n'):
                line = line.strip()
                # Prioritize 'mL' for Volume summaries, skip duplicates
                if 'mL' in line and re.search(r'\d+\.\d+', line) and len(line) < 150:
                    m_text = line.lstrip('•').lstrip('+').lstrip('-').lstrip('*').strip()
                    if m_text not in [m["Metric"] for m in self.measurements]:
                        self.measurements.append({"Metric": m_text})
        
        # Limit to top 6 clear metrics to prevent table "destruction"
        if len(self.measurements) > 6:
            self.measurements = self.measurements[:6]

        # 4. Differential
        diff_match = re.search(f"{differential_regex}(.*?)(?={impression_regex}|$)", self.raw_text, re.DOTALL | re.IGNORECASE)
        if diff_match:
            lines = diff_match.group(1).strip().split('\n')
            self.differential = [l.strip().lstrip('•').lstrip('+').lstrip('-').lstrip('*').strip() for l in lines if l.strip()]

        # 5. Impression
        impression_match = re.search(f"{impression_regex}(.*)", self.raw_text, re.DOTALL | re.IGNORECASE)
        if impression_match:
            self.impression = impression_match.group(1).strip()

def render_clinical_whiteboard(st, report: StructuredReport):
    """
    Renders the clinical whiteboard using 'st.components.v1.html' for guaranteed rendering.
    This creates an isolated sandbox, solving all raw HTML text-leakage problems.
    """
    whiteboard_css = """
    <style>
    body { background-color: transparent; margin: 0; padding: 10px; }
    .whiteboard-container {
        background-color: #FFFFFF;
        color: #1A202C !important;
        padding: 40px 60px;
        border-radius: 8px;
        border: 2px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        margin: 0 auto;
        width: calc(100% - 20px);
        box-sizing: border-box;
    }
    .whiteboard-header {
        border-bottom: 3px solid #3182CE;
        padding-bottom: 15px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .whiteboard-title { color: #2B6CB0 !important; font-size: 22px; font-weight: 800; letter-spacing: 1px; margin: 0; }
    .whiteboard-section-title {
        color: #2D3748 !important;
        font-size: 15px;
        font-weight: 700;
        text-transform: uppercase;
        border-bottom: 1px solid #EDF2F7;
        padding-bottom: 5px;
        margin: 25px 0 10px 0;
        letter-spacing: 0.5px;
    }
    .whiteboard-content { font-size: 14px; color: #4A5568 !important; margin-bottom: 8px; }
    .whiteboard-footer {
        margin-top: 50px;
        padding-top: 15px;
        border-top: 1px solid #E2E8F0;
        font-size: 10px;
        color: #718096 !important;
        text-align: center;
    }
    .measurement-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    .measurement-table td { padding: 10px; border-bottom: 1px solid #F7FAFC; color: #2C5282; font-weight: 600; font-size: 13px; }
    </style>
    """
    
    # 1. Content Prebuild
    findings_html = ""
    for f in report.findings:
        findings_html += f"<div class='whiteboard-content'>• {f}</div>"
    if not findings_html:
        findings_html = "<div class='whiteboard-content'>No itemized findings available.</div>"
    
    measurements_html = ""
    if not report.measurements:
        measurements_html = "<div class='whiteboard-content'>No quantitative metrics identified.</div>"
    else:
        measurements_html = "<table class='measurement-table'>"
        for m in report.measurements:
            measurements_html += f"<tr><td>✓ {m['Metric']}</td></tr>"
        measurements_html += "</table>"
        
    impression_val = report.impression if report.impression else "No qualitative analysis provided."
    
    # 2. Unified HTML Block
    final_html = f"""
    <html>
    <head>{whiteboard_css}</head>
    <body>
    <div class="whiteboard-container">
        <div class="whiteboard-header">
            <div class="whiteboard-title">🧠 NEUROX RADIOLOGY REPORT</div>
            <div style="text-align: right; font-size: 11px; color: #718096; font-weight: 500;">
                DATE: {report.date}<br>
                ID: NX-{datetime.now().strftime('%H%M%S')}
            </div>
        </div>
        
        <div class='whiteboard-section-title'>I. Clinical Findings</div>
        {findings_html}
        
        <div class='whiteboard-section-title'>II. Analytical Measurements</div>
        {measurements_html}
        
        <div class='whiteboard-section-title'>III. Impression</div>
        <div class='whiteboard-content' style='background: #F7FAFC; padding: 15px; border-radius: 4px; border-left: 5px solid #3182CE; font-style: italic; color: #2D3748 !important;'>
            {impression_val}
        </div>
        
        <div class="whiteboard-footer">
            COMPUTATIONAL RADIOLOGY INFRASTRUCTURE: NEUROX MULTI-GENAI ENGINE V2.2<br>
            <strong>RESEARCH AND EDUCATIONAL USE ONLY - NOT FOR CLINICAL DIAGNOSIS</strong>
        </div>
    </div>
    </body>
    </html>
    """
    
    # Render using the isolated HTML Component for 100% browser rendering stability
    import streamlit.components.v1 as components
    components.html(final_html, height=850, scrolling=True)

def generate_structured_pdf(report: StructuredReport) -> bytes:
    """
    Professional PDF generation matching the in-app view.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    style_h1 = ParagraphStyle('ReportHeader', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#2B6CB0"), spaceAfter=12)
    style_h2 = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2D3748"), borderPadding=2, spaceBefore=10, spaceAfter=10)
    style_body = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=11, leading=14)
    style_impression = ParagraphStyle('Impression', parent=styles['BodyText'], fontSize=11, leading=14, leftIndent=10, borderPadding=10, backColor=colors.HexColor("#F7FAFC"))
    style_footer = ParagraphStyle('Footer', parent=styles['BodyText'], fontSize=9, textColor=colors.grey, alignment=1)

    elements = []
    elements.append(Paragraph("NEUROX RADIOLOGY REPORT", style_h1))
    elements.append(Paragraph(f"<b>DATE:</b> {report.date} &nbsp;&nbsp; <b>ID:</b> NX-{datetime.now().strftime('%H%M%S')}", style_body))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("I. CLINICAL FINDINGS", style_h2))
    for f in report.findings:
        elements.append(Paragraph(f"• {f}", style_body))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("II. ANALYTICAL MEASUREMENTS", style_h2))
    if not report.measurements:
        elements.append(Paragraph("No significant localized pathology detected.", style_body))
    else:
        data = [[m['Metric']] for m in report.measurements]
        t = Table(data, colWidths=[450])
        t.setStyle(TableStyle([
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#2B6CB0")),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0"))
        ]))
        elements.append(t)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("III. IMPRESSION", style_h2))
    elements.append(Paragraph(report.impression if report.impression else "No qualitative analysis.", style_impression))
    
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("RESEARCH AND EDUCATIONAL USE ONLY - NOT FOR CLINICAL DIAGNOSIS", style_footer))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
