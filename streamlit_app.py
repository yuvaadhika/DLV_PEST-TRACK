"""
================================================================================
PestTrack AI - Streamlit Web Application
Intelligent Pest Infestation Monitoring & Crop Damage Analysis System
Deep Learning for Vision (DLV) Project
================================================================================
"""

import os
import sys

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
from utils.image_utils import load_image_cv, create_side_by_side_comparison
from utils.pdf_generator import generate_pest_report

# Page Configuration
st.set_page_config(
    page_title="PestTrack AI | Crop Damage & Infestation Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E392A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A6B53;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F4F9F4;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2D6A4F;
        margin-bottom: 10px;
    }
    .severity-low {
        color: #2D6A4F;
        font-weight: bold;
    }
    .severity-med {
        color: #E76F51;
        font-weight: bold;
    }
    .severity-high {
        color: #D90429;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model Loading
@st.cache_resource(show_spinner="Loading AI Vision Models (YOLOv8, CNN, U-Net)...")
def load_models():
    detector = YOLOPestDetector()
    classifier = PestClassifier()
    segmentor = LeafDamageSegmentor()
    return detector, classifier, segmentor

# Load Models
try:
    detector, classifier, segmentor = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error initializing AI models: {e}")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=64)
st.sidebar.title("🌿 PestTrack AI")
st.sidebar.markdown("**Intelligent Vision System**")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Settings")
conf_thresh = st.sidebar.slider("YOLO Detection Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Quick Test Samples")
sample_options = [
    "None (Upload my own)",
    "Sample: Damaged Leaf (High Infestation)",
    "Sample: Healthy Leaf",
    "Sample: Low Damage Leaf",
    "Sample: Aphids",
    "Sample: Fall Armyworm",
    "Sample: Beetle",
    "Sample: Whitefly"
]
selected_sample = st.sidebar.selectbox("Choose a sample to test instantly:", sample_options)

sample_map = {
    "Sample: Damaged Leaf (High Infestation)": "sample_data/leaves/high_infestation_crop.jpg",
    "Sample: Healthy Leaf": "sample_data/leaves/healthy_crop_leaf.jpg",
    "Sample: Low Damage Leaf": "sample_data/leaves/low_damage_leaf.jpg",
    "Sample: Aphids": "sample_data/pests/aphids_sample.jpg",
    "Sample: Fall Armyworm": "sample_data/pests/armyworm_sample.jpg",
    "Sample: Beetle": "sample_data/pests/beetle_sample.jpg",
    "Sample: Whitefly": "sample_data/pests/whitefly_sample.jpg",
}

st.sidebar.markdown("---")
st.sidebar.info("💡 **Project:** DLV (Deep Learning for Vision)\n\n"
               "**Architecture:**\n"
               "• YOLOv8 (Pest Detection)\n"
               "• 4-Stage CNN (Species Classification)\n"
               "• PyTorch U-Net (Damage Segmentation)")

# Header
st.markdown('<div class="main-header">🌿 PestTrack AI: Infestation Monitoring & Damage Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Model Deep Learning Vision System for Precision Agriculture</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Full AI Pipeline",
    "🎯 YOLO Detection",
    "🦠 CNN Classification",
    "🍃 U-Net Segmentation",
    "ℹ️ About Architecture"
])

# Helper to get image from sample or uploader
def get_input_image(uploader_key: str):
    image_file = st.file_uploader("Upload Crop/Pest Image (JPG, PNG)", type=["jpg", "jpeg", "png"], key=uploader_key)
    
    if image_file is not None:
        pil_img = Image.open(image_file).convert("RGB")
        return pil_img, image_file.name
    elif selected_sample != "None (Upload my own)" and selected_sample in sample_map:
        sample_path = sample_map[selected_sample]
        if os.path.exists(sample_path):
            pil_img = Image.open(sample_path).convert("RGB")
            return pil_img, os.path.basename(sample_path)
    return None, None

# ------------------------------------------------------------------------------
# TAB 1: FULL PIPELINE
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Automated Multi-Model Pipeline")
    st.write("Runs YOLOv8 Detection, PyTorch CNN Classification, and U-Net Segmentation simultaneously.")
    
    pil_img, img_name = get_input_image("full_pipeline_upload")
    
    if pil_img is not None:
        col_btn, _ = st.columns([1, 4])
        if col_btn.button("🚀 Run Complete Analysis", key="btn_run_full", type="primary"):
            with st.spinner("Analyzing with all 3 Deep Learning models..."):
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
                    sev_class = "severity-high"
                    yield_loss = "25% - 40%"
                elif (3 <= pest_count <= 7) or (10.0 <= damage_pct <= 30.0):
                    severity = "Medium"
                    sev_class = "severity-med"
                    yield_loss = "10% - 25%"
                else:
                    severity = "Low"
                    sev_class = "severity-low"
                    yield_loss = "< 10%"

            # Metrics
            st.markdown("### 📊 Analysis Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🐛 Pest Count", f"{pest_count}")
            m2.metric("🏷️ Primary Species", f"{species}")
            m3.metric("🎯 Confidence", f"{species_conf:.1%}")
            m4.metric("🍃 Leaf Damage", f"{damage_pct:.1f}%")
            m5.metric("⚠️ Severity", severity)

            st.write(f"📉 **Estimated Potential Yield Loss Range:** `{yield_loss}`")

            # Visual Comparison
            st.markdown("### 🖼️ Visual Multi-Model Outputs")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(pil_img, caption="1. Original Crop Image", use_container_width=True)
            with c2:
                st.image(det_rgb, caption=f"2. YOLOv8 Detection ({pest_count} found)", use_container_width=True)
            with c3:
                st.image(seg_overlay_rgb, caption=f"3. U-Net Segmentation ({damage_pct:.1f}% damaged)", use_container_width=True)

            # Top 3 predictions chart
            st.markdown("### 📈 Species Probability Distribution")
            chart_data = {item["species"]: item["probability"] * 100 for item in top_preds}
            st.bar_chart(chart_data)

            # PDF Report Export
            st.markdown("### 📄 Export Official Agronomist Report")
            try:
                # Save temp comparison for pdf
                comp_img = create_side_by_side_comparison(cv_img, damage_mask, seg_res["overlay"], damage_pct)
                temp_dir = os.path.join(config.DATA_DIR, "temp_web_reports")
                os.makedirs(temp_dir, exist_ok=True)
                temp_comp_path = os.path.join(temp_dir, f"comp_{img_name}")
                cv2.imwrite(temp_comp_path, comp_img)
                
                pdf_path = os.path.join(temp_dir, f"PestTrack_Report_{img_name}.pdf")
                generate_pest_report(
                    output_pdf_path=pdf_path,
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
                    result_image_path=temp_comp_path
                )
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Full PDF Advisory Report",
                    data=pdf_bytes,
                    file_name=f"PestTrack_Report_{img_name}.pdf",
                    mime="application/pdf"
                )
            except Exception as pe:
                st.info(f"Report generator ready: {pe}")

    else:
        st.info("👆 Please upload an image or select a sample from the left sidebar to begin analysis.")

# ------------------------------------------------------------------------------
# TAB 2: YOLO DETECTION ONLY
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🎯 YOLOv8 Object Detection & Pest Counting")
    pil_img_det, name_det = get_input_image("yolo_tab_upload")
    if pil_img_det:
        if st.button("Detect Pests with YOLOv8", type="primary"):
            cv_img = cv2.cvtColor(np.array(pil_img_det), cv2.COLOR_RGB2BGR)
            detections, annotated = detector.detect_pests(cv_img, conf_threshold=conf_thresh)
            ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            st.success(f"Detected {len(detections)} pests in image.")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.image(ann_rgb, caption=f"YOLO Annotations ({len(detections)} pests)", use_container_width=True)
            with col_b:
                st.markdown("**Detected Bounding Boxes:**")
                if detections:
                    for idx, d in enumerate(detections, 1):
                        st.write(f"**Pest #{idx}:** `{d['class_name']}` (Conf: {d['confidence']:.2f})")
                else:
                    st.write("No pests detected above threshold.")

# ------------------------------------------------------------------------------
# TAB 3: CNN CLASSIFIER ONLY
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🦠 PyTorch CNN Species Classification")
    pil_img_cls, name_cls = get_input_image("cnn_tab_upload")
    if pil_img_cls:
        if st.button("Classify Pest Species", type="primary"):
            cls_out = classifier.predict(pil_img_cls)
            st.success(f"**Predicted Species:** {cls_out['predicted_species']} ({cls_out['confidence']:.1%})")
            
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.image(pil_img_cls, caption="Analyzed Image", use_container_width=True)
            with c_right:
                st.markdown("**Top Probabilities:**")
                for p in cls_out["top_predictions"]:
                    st.write(f"• **{p['species']}:** {p['probability']:.1%}")
                chart_data = {item["species"]: item["probability"] * 100 for item in cls_out["top_predictions"]}
                st.bar_chart(chart_data)

# ------------------------------------------------------------------------------
# TAB 4: U-NET SEGMENTATION ONLY
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🍃 PyTorch U-Net Leaf Damage Segmentation")
    pil_img_seg, name_seg = get_input_image("unet_tab_upload")
    if pil_img_seg:
        if st.button("Segment Damaged Leaf Tissue", type="primary"):
            cv_img = cv2.cvtColor(np.array(pil_img_seg), cv2.COLOR_RGB2BGR)
            seg_out = segmentor.segment_leaf(cv_img)
            
            dmg = seg_out["damage_percentage"]
            st.markdown(f"**Calculated Leaf Damage Percentage:** `{dmg:.2f}%`")
            
            c_o, c_m, c_v = st.columns(3)
            with c_o:
                st.image(pil_img_seg, caption="Original Leaf", use_container_width=True)
            with c_m:
                st.image(seg_out["mask"], caption="Binary Damage Mask", use_container_width=True)
            with c_v:
                st.image(cv2.cvtColor(seg_out["overlay"], cv2.COLOR_BGR2RGB), caption="Damage Overlay (Red Highlight)", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: ARCHITECTURE
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("ℹ️ Deep Learning for Vision (DLV) Architecture")
    st.markdown("""
    ### System Architecture & Pipeline:
    1. **YOLOv8 Detection:** Localizes insect pest instances and computes density count.
    2. **Custom 4-Stage CNN:** 10-class species classifier identifying specific crop pests:
       *Aphids, Armyworm, Beetle, Bollworm, Grasshopper, Leafhopper, Mealybug, Mites, Stem Borer, Whitefly*.
    3. **PyTorch U-Net:** Semantic segmentation model segmenting necrotic, chlorotic, and damaged leaf area from healthy green foliage.
    4. **Infestation Severity Engine:** Tri-level agricultural threshold classification (Low / Medium / High) with potential yield loss estimation.
    5. **Automated PDF Engine:** Formulates formal inspection reports with metadata and legal agronomic disclaimers.
    """)
    st.info(f"⚖️ **Disclaimer:** {config.DISCLAIMER_TEXT}")
