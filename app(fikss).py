import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Penilaian Kualitas Pisang (Non-AI)",
    layout="wide"
)

# ============================================================
# STYLE / UI
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {background: linear-gradient(180deg, #FFF8E1, #FFF3C4) !important;}
div[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #FFF8E1, #FFF3C4) !important;}
.main {background: linear-gradient(180deg, #FFF8E1, #FFF3C4) !important;}
header[data-testid="stHeader"] {background: linear-gradient(180deg, #FFF8E1, #FFF3C4) !important;}
section.main > div {background: transparent !important;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #FFE082, #FFD54F) !important; box-shadow: inset -4px 0 12px rgba(0,0,0,0.15);}
section[data-testid="stSidebar"] * {color: #4E342E !important; font-weight: 500;}
section[data-testid="stSidebar"] h1, h2, h3 {color: #3E2723 !important; font-weight: 700;}
section[data-testid="stSidebar"] button {background: linear-gradient(135deg, #FFCA28, #FFB300) !important; color: #3E2723 !important; border-radius: 10px; border: none; font-weight: 600;}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"], section[data-testid="stSidebar"] div[data-testid="stCameraInput"] {background: rgba(255,255,255,0.45); border-radius: 12px; padding: 10px;}
.card {background: linear-gradient(135deg, #FFFDE7, #FFF9C4); padding: 25px; border-radius: 18px; box-shadow: 0px 8px 20px rgba(0,0,0,0.12); margin-bottom: 25px;}
.step-box {background: linear-gradient(135deg, #FFF176, #FFD54F); padding: 18px; border-left: 6px solid #F1C40F; border-radius: 12px; font-weight: 600;}
.footer {text-align: center; color: #6D4C41; font-size: 13px; margin-top: 40px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<h1 style='text-align:center; color:#2E7D32;'>🍌 Sistem Penilaian Kualitas Buah Pisang</h1>
<p style='text-align:center; font-size:18px; color:#5D4037;'>Berbasis Segmentasi Warna & Analisis Tekstur (Non-AI)</p>
<hr>
""", unsafe_allow_html=True)

# ============================================================
# FUNGSI GLCM MANUAL
# ============================================================
def glcm_features_manual(gray):
    gray = gray // 16
    glcm = np.zeros((16, 16), dtype=np.float32)
    for i in range(gray.shape[0] - 1):
        for j in range(gray.shape[1] - 1):
            glcm[gray[i, j], gray[i, j + 1]] += 1
    glcm /= np.sum(glcm) + 1e-6
    contrast = 0
    homogeneity = 0
    for i in range(16):
        for j in range(16):
            contrast += (i - j) ** 2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
    return contrast, homogeneity

# ============================================================
# EKSTRAKSI FITUR
# ============================================================
def ekstraksi_fitur(img_rgb, thresh_s, thresh_v, thresh_busuk_v):
    img_rgb = cv2.resize(img_rgb, (400, 400))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask_pisang = ((s > thresh_s) & (v > thresh_v)).astype(np.uint8)
    area_total = np.sum(mask_pisang)
    mask_busuk = ((v < thresh_busuk_v) & (s > 20) & ((h < 20) | (h > 160))).astype(np.uint8)
    area_busuk = np.sum(mask_busuk & mask_pisang)
    persen_busuk = (area_busuk / area_total * 100) if area_total > 0 else 0
    h_mean = np.mean(h[mask_pisang == 1]) if area_total > 0 else 0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    contrast, homogeneity = glcm_features_manual(gray)
    return {
        "h_mean": h_mean,
        "persen_busuk": persen_busuk,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "img": img_rgb,
        "mask_pisang": mask_pisang * 255,
        "mask_busuk": mask_busuk * 255,
        "gray": gray
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.caption("Atur parameter sesuai kondisi pencahayaan")
    mode_input = st.radio("Mode Input", ["Upload Gambar", "Kamera"])
    uploaded_file = None
    camera_image = None
    if mode_input == "Upload Gambar":
        uploaded_file = st.file_uploader("Upload Citra Pisang", type=["jpg", "jpeg", "png"])
    else:
        camera_image = st.camera_input("Ambil Gambar dari Kamera")
    st.divider()
    st.subheader("🎚️ Parameter Tuning")
    thresh_s = st.slider("Minimum Saturation", 10, 80, 30)
    thresh_v = st.slider("Minimum Value", 10, 80, 30)
    thresh_busuk_v = st.slider("Value Maks Area Busuk", 60, 160, 120)
    batas_busuk = st.slider("Ambang BUSUK (%)", 20, 50, 30)
    batas_hampir_busuk = st.slider("Ambang HAMPIR BUSUK (%)", 5, 20, 10)

# ============================================================
# PROSES UTAMA
# ============================================================
if uploaded_file or camera_image:
    image = Image.open(uploaded_file if uploaded_file else camera_image).convert("RGB")
    img_rgb = np.array(image)
    data = ekstraksi_fitur(img_rgb, thresh_s, thresh_v, thresh_busuk_v)

    h_mean = data["h_mean"]
    persen_busuk = data["persen_busuk"]
    contrast = data["contrast"]
    homogeneity = data["homogeneity"]

    if persen_busuk >= batas_busuk:
        keputusan, warna = "BUSUK", "#C0392B"
    elif batas_hampir_busuk <= persen_busuk < batas_busuk:
        keputusan, warna = "HAMPIR BUSUK", "#E67E22"
    elif h_mean >= 40:
        keputusan, warna = "MENTAH", "#2E7D32"
    elif 20 <= h_mean < 40:
        keputusan, warna = "MATANG SEMPURNA", "#FBC02D"
    else:
        keputusan, warna = "TIDAK TERDETEKSI", "#616161"

    st.markdown("""
    <div class="step-box">🔍 <b>Alur Analisis Sistem</b><br>
    Input → Segmentasi → Analisis Permukaan → Ekstraksi Fitur → Keputusan
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🖼️ Visualisasi Proses")
    c1, c2, c3, c4 = st.columns(4)
    c1.image(data["img"], caption="Citra Asli")
    c2.image(data["mask_pisang"], caption="Mask Pisang")
    c3.image(data["mask_busuk"], caption="Mask Busuk")
    c4.image(data["gray"], caption="Grayscale")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Nilai Fitur")
    df = pd.DataFrame({
        "Fitur": ["Hue Rata-rata", "Area Busuk (%)", "Contrast", "Homogeneity"],
        "Nilai": [f"{h_mean:.2f}", f"{persen_busuk:.2f}", f"{contrast:.4f}", f"{homogeneity:.4f}"]
    })
    st.table(df)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("📌 Keputusan Akhir")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {warna}, rgba(0,0,0,0.85));
        padding:30px;
        border-radius:18px;
        text-align:center;
        box-shadow:0px 6px 18px rgba(0,0,0,0.3);">
        <h1 style="color:white;">{keputusan}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # DOWNLOAD PDF
    # ============================================================
    def create_pdf(dataframe, keputusan, warna, images):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Hasil Analisis Kualitas Buah Pisang", styles['Title']))
        elements.append(Spacer(1, 12))

        # Tambahkan Tabel Fitur
        tbl_data = [dataframe.columns.tolist()] + dataframe.values.tolist()
        tbl = Table(tbl_data)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightyellow),
            ('TEXTCOLOR',(0,0),(-1,0),colors.black),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 12))

        # Keputusan Akhir
        elements.append(Paragraph(f"Keputusan Akhir: {keputusan}", styles['Heading2']))
        elements.append(Spacer(1, 12))

        # Tambahkan semua gambar
        for img in images:
            buf = BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            rl_img = RLImage(buf, width=200, height=200)
            elements.append(rl_img)
            elements.append(Spacer(1, 12))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pil_images = [Image.fromarray(data["img"]), Image.fromarray(data["mask_pisang"]),
                  Image.fromarray(data["mask_busuk"]), Image.fromarray(data["gray"])]

    pdf_buffer = create_pdf(df, keputusan, warna, pil_images)
    st.download_button(
        label="⬇️ Download PDF Hasil Analisis",
        data=pdf_buffer,
        file_name="hasil_analisis_pisang.pdf",
        mime="application/pdf"
    )

else:
    st.info("Silakan upload citra pisang atau gunakan kamera untuk memulai.")

st.markdown("""
<div class="footer">
Sistem Penilaian Kualitas Buah Pisang | Image Processing | Rule-Based Non-AI<br>
K.9 (ARINA • HAFIDZ • INTAN • MIRNA)
</div>
""", unsafe_allow_html=True)
