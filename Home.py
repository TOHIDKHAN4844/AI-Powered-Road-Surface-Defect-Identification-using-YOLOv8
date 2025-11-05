import streamlit as st
from PIL import Image

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection App",
    page_icon="🛣️",
    layout="wide"
)

# ------------------------------------------------
# TITLE SECTION
# ------------------------------------------------
st.title("🛣️ Road Damage Detection System")
st.subheader("Empowering Safer Roads through AI-Powered Detection")
st.markdown("---")

# ------------------------------------------------
# INTRODUCTION
# ------------------------------------------------
st.markdown("""
### 🌍 Overview
The **Road Damage Detection System** is an AI-powered solution built with **YOLOv8-Small**, trained on the **CRDDC 2022 (Japan + India)** dataset.
It automatically detects and classifies common types of road surface damage, enabling faster, data-driven road maintenance.
""")

# ------------------------------------------------
# ROAD DAMAGE TYPES
# ------------------------------------------------
st.markdown("## 🧩 Types of Road Damages Detected by the Model")

col1, col2 = st.columns(2)
with col1:
    st.image("resource/Alligator-Cracking-1024x576.jpg", caption="🕸️ Alligator Crack", width=500)
    st.image("resource/longtidunal-cracks-1024x576.jpg", caption="↕️ Longitudinal Crack", width=500)
with col2:
    st.image("resource/Transverse-Cracking-1024x576.jpg", caption="↔️ Transverse Crack", width=500)
    st.image("resource/pothole-1024x576.jpg", caption="🕳️ Pothole", width=500)

st.markdown("""
Each type of crack has unique visual characteristics:
- **Alligator Cracks:** Networked cracking pattern resembling reptile scales.
- **Longitudinal Cracks:** Run parallel to the road’s centerline.
- **Transverse Cracks:** Run perpendicular to the traffic direction.
- **Potholes:** Depressions or holes formed due to water and load stress.
""")

# ------------------------------------------------
# MODEL PERFORMANCE SECTION
# ------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Model Training & Evaluation Summary")

col3, col4 = st.columns(2)
with col3:
    st.image("resource/confusion_matrix.png", caption="Confusion Matrix", width=500)
with col4:
    st.image("resource/PR_curve.png", caption="Precision-Recall Curve", width=500)

st.markdown("""
**Model Configuration**
- Model: YOLOv8-Small  
- Dataset: CRDDC 2022 (Japan + India)  
- Epochs: 100  
- Image Size: 640×640  
- Batch Size: 16  
- Optimizer: SGD  
- Loss: Objectness + Classification + Box Regression  

**Performance Metrics (Validation):**
| Metric | Value |
|:-------|:------:|
| mAP@0.5 | 0.82 |
| mAP@0.5:0.95 | 0.67 |
| Precision | 0.84 |
| Recall | 0.79 |
| F1-score | 0.81 |
""")

# ------------------------------------------------
# VALIDATION EXAMPLES
# ------------------------------------------------
st.markdown("## 🖼️ Example Results on Validation Set")

col5, col6 = st.columns(2)
with col5:
    st.image("resource/val_batch2_labels.jpg", caption="Ground Truth Labels", width=500)
with col6:
    st.image("resource/val_batch2_pred.jpg", caption="Predicted Results", width=500)

st.caption("The model performs robustly across varied lighting and texture conditions.")

# ------------------------------------------------
# KEY FEATURES SECTION
# ------------------------------------------------
st.markdown("""
---
## 💡 Key Highlights
-  **Fast inference:** YOLOv8 achieves high FPS with accuracy.  
-  **Confidence visualization:** Each bounding box shows prediction confidence.  
-  **Multi-region generalization:** Trained on Japan & India datasets.  
-  **End-to-end pipeline:** Preprocessing → Training → Evaluation → Deployment.

---
## ⚙️ Technical Stack
| Component | Description |
|:----------|:------------|
| Framework | Streamlit |
| Model | YOLOv8-Small |
| Dataset | CRDDC 2022 |
| Language | Python |
| Libraries | Ultralytics, OpenCV, Torch, Pillow |
| Deployment | Local / Cloud Streamlit |

---
##  How to Use
1. Open **Image Detection** from the sidebar.  
2. Upload a road image (JPG/PNG).  
3. Adjust the confidence threshold if needed.  
4. View and download annotated prediction results instantly.
""")

# ------------------------------------------------
# CREDITS & REFERENCES
# ------------------------------------------------
st.markdown("""
---
##  Acknowledgements
Special thanks to **Ultralytics**, **CRDDC 2022 dataset contributors**, and the **open-source AI community**.  

📧 **Contact:** [tohid3707gmail.com](mailto:tohid3707gmail.com)  
🌐 **GitHub:** [RoadDamageDetection](https://github.com/tohidkhan4844)


---
**“Better Roads. Safer Journeys.”**
""")
