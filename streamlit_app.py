"""
================================================================================
PestTrack AI - Web Application (Matches Desktop GUI Exactly)
Theme: Modern Agriculture Green & White Interface
================================================================================
"""

import os
import sys
import datetime

# Ensure local config directories are configured before any imports
os.environ["YOLO_CONFIG_DIR"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), ".ultralytics")
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), ".matplotlib")

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import config
from ai_models.yolo_detector import YOLOPestDetector
from ai_models.cnn_classifier import PestClassifier
from ai_models.unet_segmentor import LeafDamageSegmentor
from database.db_manager import DatabaseManager
from utils.image_utils import load_image_cv, create_side_by_side_comparison
from utils.pdf_generator import generate_pdf_report, generate_pest_report

# Page Configuration
st.set_page_config(
    page_title="PestTrack AI - Intelligent Crop Vision",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to match CustomTkinter Green & White Theme
st.markdown("""
<style>
    /* Global Background & Typography */
    .main {
        background-color: #F4F7F5;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        border-radius: 12px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.15);
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #FFFFFF !important;
    }
    .header-subtitle {
        font-size: 1.0rem;
        color: #E8F5E9 !important;
        margin-bottom: 0px;
    }

    /* Metric Cards */
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #555555;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1C2826;
    }

    /* Severity Badges */
    .badge-low {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }
    .badge-med {
        background-color: #FFF3E0;
        color: #F57C00;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FFCC80;
    }
    .badge-high {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #EF9A9A;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #E8F5E9;
    }
    .sidebar-brand {
        font-size: 1.5rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 2px;
    }
    .sidebar-tagline {
        font-size: 0.85rem;
        color: #607D8B;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Cache AI Models
@st.cache_resource(show_spinner="Initializing AI Vision Models (YOLOv8, CNN, U-Net)...")
def load_all_models():
    detector = YOLOPestDetector()
    classifier = PestClassifier()
    segmentor = LeafDamageSegmentor()
    db = DatabaseManager()
    return detector, classifier, segmentor, db

detector, classifier, segmentor, db = load_all_models()

# ------------------------------------------------------------------------------
# SIDEBAR (Identical to CustomTkinter Navigation Sidebar)
# ------------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-brand">🌱 PestTrack AI</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-tagline">Intelligent Crop Vision System</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation radio mimicking the desktop buttons
nav_choice = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠  Home",
        "🔍  Pest Detection",
        "🐛  Pest Species",
        "🍃  Damage Segmentation",
        "📊  Analysis History",
        "📄  PDF Reports",
        "ℹ️  About Project"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Preloaded Test Samples")
sample_choice = st.sidebar.selectbox(
    "Test instant sample image:",
    [
        "None (Upload my own)",
        "Sample 1: High Infestation Leaf",
        "Sample 2: Healthy Crop Leaf",
        "Sample 3: Low Damage Leaf",
        "Sample 4: Aphids Infestation",
        "Sample 5: Armyworm Pest",
        "Sample 6: Beetle Pest",
        "Sample 7: Whitefly Crop"
    ]
)

sample_dict = {
    "Sample 1: High Infestation Leaf": "sample_data/leaves/high_infestation_crop.jpg",
    "Sample 2: Healthy Crop Leaf": "sample_data/leaves/healthy_crop_leaf.jpg",
    "Sample 3: Low Damage Leaf": "sample_data/leaves/low_damage_leaf.jpg",
    "Sample 4: Aphids Infestation": "sample_data/pests/aphids_sample.jpg",
    "Sample 5: Armyworm Pest": "sample_data/pests/armyworm_sample.jpg",
    "Sample 6: Beetle Pest": "sample_data/pests/beetle_sample.jpg",
    "Sample 7: Whitefly Crop": "sample_data/pests/whitefly_sample.jpg",
}

st.sidebar.markdown("---")
st.sidebar.caption("DLV Academic Project | 3rd Year B.Tech AIML")

def get_active_image(key: str):
    uploader = st.file_uploader("Choose a crop image (JPG, PNG)", type=["jpg", "jpeg", "png"], key=key)
    if uploader is not None:
        return Image.open(uploader).convert("RGB"), uploader.name
    elif sample_choice != "None (Upload my own)" and sample_choice in sample_dict:
        path = sample_dict[sample_choice]
        if os.path.exists(path):
            return Image.open(path).convert("RGB"), os.path.basename(path)
    return None, None

# ==============================================================================
# VIEW 1: 🏠 HOME DASHBOARD
# ==============================================================================
if "Home" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🌾 Welcome to PestTrack AI Dashboard</div>
        <div class="header-subtitle">Real-Time Pest Detection (YOLO), Species Classification (CNN), & Leaf Damage Segmentation (U-Net)</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Metric Cards (Matching desktop HomeView)
    stats = db.get_dashboard_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #1976D2;">
            <div class="kpi-title" style="color: #1976D2;">📸 TOTAL ANALYZED</div>
            <div class="kpi-value">{stats['total_analyzed']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #388E3C;">
            <div class="kpi-title" style="color: #388E3C;">🐛 PESTS DETECTED</div>
            <div class="kpi-value">{stats['total_pests']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #D32F2F;">
            <div class="kpi-title" style="color: #D32F2F;">⚠️ HIGH SEVERITY CASES</div>
            <div class="kpi-value">{stats['high_severity_cases']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #7B1FA2;">
            <div class="kpi-title" style="color: #7B1FA2;">🤖 AI MODELS STATUS</div>
            <div class="kpi-value">3/3 Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚡ Quick End-to-End Analysis")
    st.write("Upload an image below or pick a sample from the left sidebar to execute the full multi-model pipeline.")

    pil_img, img_name = get_active_image("home_upload")

    if pil_img is not None:
        col_img, col_act = st.columns([1, 1])
        with col_img:
            st.image(pil_img, caption=f"Selected Input: {img_name}", use_container_width=True)
        with col_act:
            st.write("Ready to analyze with YOLOv8, CNN 10-Class Classifier, and U-Net Segmentor.")
            if st.button("🚀 Run Complete Multi-Model Pipeline", type="primary", use_container_width=True):
                with st.spinner("Processing deep learning vision pipeline..."):
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # 1. Detection
                    detections, det_annotated = detector.detect_pests(cv_img, conf_threshold=conf_thresh)
                    pest_count = len(detections)
                    det_rgb = cv2.cvtColor(det_annotated, cv2.COLOR_BGR2RGB)

                    # 2. Classification
                    if detections:
                        top_box = sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]["bbox"]
                        x1, y1, x2, y2 = [int(v) for v in top_box]
                        h, w = cv_img.shape[:2]
                        crop_pil = pil_img.crop((max(0, x1-10), max(0, y1-10), min(w, x2+10), min(h, y2+10)))
                        cls_res = classifier.predict(crop_pil)
                    else:
                        cls_res = classifier.predict(pil_img)

                    species = cls_res["predicted_species"]
                    species_conf = cls_res["confidence"]
                    top_preds = cls_res["top_predictions"]

                    # 3. Segmentation
                    seg_res = segmentor.segment_leaf(cv_img)
                    damage_pct = seg_res["damage_percentage"]
                    seg_overlay_rgb = cv2.cvtColor(seg_res["overlay"], cv2.COLOR_BGR2RGB)
                    damage_mask = seg_res["mask"]

                    # Severity
                    if pest_count >= 8 or damage_pct > 30.0:
                        severity = "High"
                        badge_html = '<span class="badge-high">⚠️ HIGH SEVERITY</span>'
                        yield_range = "25% - 40%"
                    elif (3 <= pest_count <= 7) or (10.0 <= damage_pct <= 30.0):
                        severity = "Medium"
                        badge_html = '<span class="badge-med">⚡ MEDIUM SEVERITY</span>'
                        yield_range = "10% - 25%"
                    else:
                        severity = "Low"
                        badge_html = '<span class="badge-low">✅ LOW SEVERITY</span>'
                        yield_range = "< 10%"

                    # Save record into DB
                    comp_img = create_side_by_side_comparison(cv_img, damage_mask, seg_res["overlay"], damage_pct)
                    temp_dir = os.path.join(config.DATA_DIR, "exports", "segmentations")
                    os.makedirs(temp_dir, exist_ok=True)
                    res_path = os.path.join(temp_dir, f"web_comp_{img_name}")
                    cv2.imwrite(res_path, comp_img)

                    db.insert_analysis(
                        analysis_type="full",
                        image_name=img_name,
                        original_image_path="",
                        result_image_path=res_path,
                        pest_count=pest_count,
                        primary_pest_species=species,
                        confidence=species_conf,
                        damage_percentage=damage_pct,
                        severity_level=severity,
                        notes=f"Web inspection: {pest_count} pests, {damage_pct:.1f}% damage.",
                        detections=detections
                    )

                # Display Results
                st.markdown("---")
                st.markdown("### 📊 Pipeline Inspection Results")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🐛 Pests Found", f"{pest_count}")
                m2.metric("🏷️ Primary Species", f"{species}")
                m3.metric("🎯 Confidence", f"{species_conf:.1%}")
                m4.metric("🍃 Leaf Damage", f"{damage_pct:.1f}%")

                st.markdown(f"**Infestation Severity Status:** {badge_html} &nbsp;&nbsp;|&nbsp;&nbsp; **Estimated Yield Loss:** `{yield_range}`", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # 3 Images Comparison
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    st.image(pil_img, caption="1. Original Crop Leaf", use_container_width=True)
                with ic2:
                    st.image(det_rgb, caption=f"2. YOLOv8 Detections ({pest_count} instances)", use_container_width=True)
                with ic3:
                    st.image(seg_overlay_rgb, caption=f"3. U-Net Damage Segmentation ({damage_pct:.1f}%)", use_container_width=True)

                # Species Probabilities
                st.markdown("### 📈 Species Probability Distribution")
                chart_data = {item["species"]: item["probability"] * 100 for item in top_preds}
                st.bar_chart(chart_data)

                # PDF Export
                st.markdown("### 📄 Automated PDF Advisory Report")
                rep_dir = os.path.join(config.DATA_DIR, "exports", "reports")
                os.makedirs(rep_dir, exist_ok=True)
                pdf_out = os.path.join(rep_dir, f"PestTrack_Report_{img_name}.pdf")
                generate_pest_report(
                    output_pdf_path=pdf_out,
                    analysis_data={
                        "id": 1,
                        "image_name": img_name,
                        "pest_count": pest_count,
                        "primary_pest_species": species,
                        "confidence": species_conf,
                        "damage_percentage": damage_pct,
                        "severity_level": severity,
                        "top_predictions": top_preds,
                        "detections": detections
                    },
                    result_image_path=res_path
                )
                if os.path.exists(pdf_out):
                    with open(pdf_out, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="📥 Download Official Agronomist PDF Report",
                        data=pdf_data,
                        file_name=f"PestTrack_Report_{img_name}.pdf",
                        mime="application/pdf"
                    )

# ==============================================================================
# VIEW 2: 🔍 PEST DETECTION
# ==============================================================================
elif "Detection" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🔍 Pest Detection & Counting</div>
        <div class="header-subtitle">YOLOv8 Deep Learning Object Detection with Bounding Box Localization</div>
    </div>
    """, unsafe_allow_html=True)

    pil_img, img_name = get_active_image("yolo_view_upload")
    if pil_img:
        col_btn, _ = st.columns([1, 3])
        if col_btn.button("Detect Pests with YOLOv8", type="primary"):
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            detections, annotated = detector.detect_pests(cv_img, conf_threshold=conf_thresh)
            ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.success(f"YOLOv8 successfully detected {len(detections)} pests in image.")
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.image(ann_rgb, caption=f"Annotated Image ({len(detections)} instances)", use_container_width=True)
            with c_right:
                st.markdown("#### Detection Details")
                if detections:
                    for i, d in enumerate(detections, 1):
                        st.info(f"**Pest #{i}:** `{d['class_name']}`\n\n**Confidence:** {d['confidence']:.2f}")
                else:
                    st.warning("No pests found above threshold.")

# ==============================================================================
# VIEW 3: 🐛 PEST SPECIES
# ==============================================================================
elif "Species" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🐛 10-Class Pest Species Classification</div>
        <div class="header-subtitle">Custom 4-Stage PyTorch CNN with Softmax Probability Distribution</div>
    </div>
    """, unsafe_allow_html=True)

    pil_img, img_name = get_active_image("species_view_upload")
    if pil_img:
        if st.button("Classify Species", type="primary"):
            cls_out = classifier.predict(pil_img)
            st.success(f"**Identified Species:** {cls_out['predicted_species']} ({cls_out['confidence']:.1%})")

            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.image(pil_img, caption="Analyzed Image", use_container_width=True)
            with c_right:
                st.markdown("#### Top Probabilities")
                for p in cls_out["top_predictions"]:
                    st.write(f"• **{p['species']}:** {p['probability']:.1%}")
                chart_data = {item["species"]: item["probability"] * 100 for item in cls_out["top_predictions"]}
                st.bar_chart(chart_data)

# ==============================================================================
# VIEW 4: 🍃 DAMAGE SEGMENTATION
# ==============================================================================
elif "Segmentation" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🍃 Leaf Damage Semantic Segmentation</div>
        <div class="header-subtitle">PyTorch U-Net Pixel-Level Segmentation & Damage Area Ratio Calculation</div>
    </div>
    """, unsafe_allow_html=True)

    pil_img, img_name = get_active_image("seg_view_upload")
    if pil_img:
        if st.button("Segment Damaged Foliage", type="primary"):
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            seg_out = segmentor.segment_leaf(cv_img)
            dmg = seg_out["damage_percentage"]

            st.metric("🍃 Leaf Damage Percentage", f"{dmg:.2f}%")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(pil_img, caption="Original Foliage", use_container_width=True)
            with c2:
                st.image(seg_out["mask"], caption="Binary Damage Mask", use_container_width=True)
            with c3:
                st.image(cv2.cvtColor(seg_out["overlay"], cv2.COLOR_BGR2RGB), caption="Damage Overlay (Red Highlight)", use_container_width=True)

# ==============================================================================
# VIEW 5: 📊 ANALYSIS HISTORY
# ==============================================================================
elif "History" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">📊 Analysis History & Audit Trail</div>
        <div class="header-subtitle">Persistent SQLite Database Records of All Prior Pest Inspections</div>
    </div>
    """, unsafe_allow_html=True)

    records = db.get_recent_analyses(limit=50)
    if records:
        table_data = []
        for r in records:
            table_data.append({
                "ID": r["id"],
                "Timestamp": r["timestamp"],
                "Image": r["image_name"],
                "Pests Found": r["pest_count"],
                "Primary Species": r["primary_pest_species"],
                "Confidence": f"{r['confidence']:.1%}",
                "Leaf Damage %": f"{r['damage_percentage']:.1f}%",
                "Severity": r["severity_level"]
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("No historical records logged yet. Run an analysis from the Home tab to log records.")

# ==============================================================================
# VIEW 7: 📄 PDF REPORTS
# ==============================================================================
elif "Reports" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">📄 Inspection PDF Reports</div>
        <div class="header-subtitle">Automated ReportLab Advisory Reports with Tables, Visual Masks & Agronomic Notices</div>
    </div>
    """, unsafe_allow_html=True)

    records = db.get_recent_analyses(limit=10)
    if records:
        st.write("Select a saved inspection record to generate and download its formal PDF report:")
        selected_rec = st.selectbox(
            "Select Record:",
            [f"Record #{r['id']} - {r['image_name']} ({r['timestamp']}) - {r['severity_level']} Severity" for r in records]
        )
        rec_id = int(selected_rec.split(" - ")[0].replace("Record #", ""))
        rec_data = next((r for r in records if r["id"] == rec_id), None)

        if rec_data and st.button("Generate & Download PDF", type="primary"):
            rep_path = os.path.join(config.DATA_DIR, "exports", "reports", f"Report_Record_{rec_id}.pdf")
            generate_pest_report(output_pdf_path=rep_path, analysis_data=rec_data)
            if os.path.exists(rep_path):
                with open(rep_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF Document",
                        data=f.read(),
                        file_name=f"PestTrack_Record_{rec_id}.pdf",
                        mime="application/pdf"
                    )
    else:
        st.info("Analyze an image from the Home tab first to generate reports.")

# ==============================================================================
# VIEW 8: ℹ️ ABOUT PROJECT
# ==============================================================================
elif "About" in nav_choice:
    st.markdown("""
    <div class="header-card">
        <div class="header-title">ℹ️ System Architecture & DLV Specifications</div>
        <div class="header-subtitle">Deep Learning for Vision (DLV) Academic Project Architecture & Methodology</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Multi-Model Deep Learning Architecture:
    1. **Model 1: YOLOv8 Pest Object Detection & Density Counter**
       - Task: Localizes insect pests with rectangular bounding boxes.
       - Classes: Detects agricultural pests across diverse crop leaves.
       - Weight: `models/yolov8_pest.pt`

    2. **Model 2: 4-Stage PyTorch CNN Pest Species Classifier**
       - Task: Fine-grained classification across 10 agricultural species:
         *Aphids, Fall Armyworm, Beetle, Bollworm, Grasshopper, Leafhopper, Mealybug, Mites, Stem Borer, Whitefly*.
       - Weight: `models/pest_classifier_cnn.pth`

    3. **Model 3: PyTorch U-Net Leaf Damage Semantic Segmentor**
       - Task: Pixel-level segmentation distinguishing necrotic, chlorotic, and insect-chewed leaf tissue from healthy foliage.
       - Loss Function: Combined Binary Cross-Entropy + Dice Coefficient Loss.
       - Weight: `models/leaf_damage_unet.pth`

    4. **Rule-Based Infestation Severity & Yield Loss Estimation Engine**
       - High Severity: Pest count >= 8 OR Leaf Damage > 30%
       - Medium Severity: Pest count 3 to 7 OR Leaf Damage 10% to 30%
       - Low Severity: Pest count <= 2 AND Leaf Damage < 10%

    ---
    ### ⚖️ Academic & Agronomic Notice:
    """)
    st.info(f"**Disclaimer:** {config.DISCLAIMER_TEXT}")
