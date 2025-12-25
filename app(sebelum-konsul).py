import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Penilaian Kualitas Pisang (Non-AI)",
    layout="wide"
)

st.title("🍌 Penilaian Kualitas Buah Pisang")
st.markdown("**Segmentasi HSV + Analisis Tekstur GLCM (Rule-Based, Tanpa AI)**")

# ============================================================
# FUNGSI GLCM MANUAL (TANPA SKIMAGE)
# ============================================================
def glcm_features_manual(gray):
    gray = gray // 16  # kuantisasi 16 level
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
# FUNGSI UTAMA EKSTRAKSI FITUR
# ============================================================
def ekstraksi_fitur(img_rgb, threshold_gelap):
    img_rgb = cv2.resize(img_rgb, (400, 400))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)

    # Segmentasi pisang (kuning–hijau)
    lower = np.array([18, 50, 50])
    upper = np.array([40, 255, 255])
    mask_pisang = cv2.inRange(hsv, lower, upper)

    # Hue rata-rata
    F_Hue = np.mean(h[mask_pisang > 0]) if np.sum(mask_pisang) > 0 else 0

    # Area gelap
    mask_gelap = (v < threshold_gelap).astype(np.uint8)
    area_gelap = np.sum((mask_gelap == 1) & (mask_pisang > 0))
    area_total = np.sum(mask_pisang > 0)

    F_AreaGelap = (area_gelap / area_total * 100) if area_total > 0 else 0

    # Tekstur GLCM
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    F_Contrast, F_Homogeneity = glcm_features_manual(gray)

    return (
        F_Hue,
        F_AreaGelap,
        F_Contrast,
        F_Homogeneity,
        img_rgb,
        mask_pisang,
        gray,
        mask_gelap * 255
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    uploaded_file = st.file_uploader(
        "Upload Citra Pisang",
        type=["jpg", "jpeg", "png"]
    )

    threshold_gelap = st.slider(
        "Threshold Gelap (Value HSV)",
        0, 255, 40, step=5
    )

    batas_busuk = st.slider(
        "Batas Area Gelap Busuk (%)",
        5, 30, 15
    )

# ============================================================
# INISIALISASI OUTPUT
# ============================================================
KeputusanAkhir = "Menunggu Citra"
WarnaKeputusan = "#A9A9A9"
LogikaKeputusan = ["Silakan upload citra pisang."]

# ============================================================
# PROSES UTAMA
# ============================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    (
        F_Hue,
        F_AreaGelap,
        F_Contrast,
        F_Homogeneity,
        VIS_IMG,
        VIS_MASK,
        VIS_GRAY,
        VIS_BUSUK
    ) = ekstraksi_fitur(img_rgb, threshold_gelap)

    LogikaKeputusan = []

    # ========================================================
    # RULE-BASED CLASSIFICATION (TEGAS & ILMIAH)
    # ========================================================
    if F_AreaGelap >= batas_busuk:
        KeputusanAkhir = "BUSUK"
        WarnaKeputusan = "red"
        LogikaKeputusan.append(
            f"Area gelap {F_AreaGelap:.1f}% ≥ {batas_busuk}%"
        )
        LogikaKeputusan.append(
            f"Tekstur kasar (Contrast = {F_Contrast:.2f})"
        )

    elif F_Hue >= 42:
        KeputusanAkhir = "KURANG MATANG"
        WarnaKeputusan = "#CCCC00"
        LogikaKeputusan.append(
            f"Hue hijau dominan ({F_Hue:.1f})"
        )
        LogikaKeputusan.append(
            f"Area gelap rendah ({F_AreaGelap:.1f}%)"
        )

    elif 20 <= F_Hue <= 35:
        KeputusanAkhir = "MATANG"
        WarnaKeputusan = "green"
        LogikaKeputusan.append(
            f"Hue kuning stabil ({F_Hue:.1f})"
        )
        LogikaKeputusan.append(
            f"Permukaan homogen (Homogeneity = {F_Homogeneity:.2f})"
        )

    else:
        KeputusanAkhir = "KURANG MATANG"
        WarnaKeputusan = "orange"
        LogikaKeputusan.append(
            f"Hue berada di zona transisi ({F_Hue:.1f})"
        )
        LogikaKeputusan.append(
            "Belum memenuhi kriteria matang atau kurang matang"
        )

    # ========================================================
    # VISUALISASI
    # ========================================================
    st.header("🖼️ Visualisasi Proses")
    c1, c2, c3, c4 = st.columns(4)

    c1.image(VIS_IMG, caption="Citra Asli")
    c2.image(VIS_MASK, caption="Mask Segmentasi Pisang")
    c3.image(VIS_GRAY, caption="Citra Grayscale")
    c4.image(VIS_BUSUK, caption="Mask Area Gelap")

    # ========================================================
    # TABEL FITUR
    # ========================================================
    st.header("📊 Hasil Ekstraksi Fitur")

    df = pd.DataFrame({
        "Fitur": [
            "Rata-rata Hue",
            "Area Gelap (%)",
            "Contrast (GLCM)",
            "Homogeneity (GLCM)"
        ],
        "Nilai": [
            f"{F_Hue:.2f}",
            f"{F_AreaGelap:.2f}",
            f"{F_Contrast:.4f}",
            f"{F_Homogeneity:.4f}"
        ]
    })

    st.table(df.set_index("Fitur"))

# ============================================================
# OUTPUT AKHIR
# ============================================================
st.header("✅ Hasil Penilaian")

st.markdown(
    f"""
    <div style='background-color:{WarnaKeputusan};
                padding:20px;
                border-radius:10px;
                text-align:center;'>
        <h1 style='color:white;'>{KeputusanAkhir}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Logika Keputusan:")
for log in LogikaKeputusan:
    st.info(log)