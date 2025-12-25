import cv2
import numpy as np
import glob
import os
import csv

# ==============================
# FOLDER SETUP
# ==============================

input_folders = {
    "matang": "dataset/matang/",
    "mentah": "dataset/mentah/",
    "busuk": "dataset/busuk/"
}

output_mask = "output/mask/"
output_csv = "output/csv/"
output_features = "output/fitur/"

for folder in [output_mask, output_csv, output_features]:
    os.makedirs(folder, exist_ok=True)


# ==============================
# GLCM MANUAL
# ==============================

def glcm_manual(gray):
    glcm = np.zeros((256, 256), dtype=np.float64)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1] - 1):
            row = gray[i, j]
            col = gray[i, j + 1]
            glcm[row, col] += 1

    glcm = glcm / glcm.sum()

    contrast = np.sum(glcm * (np.indices((256,256))[0] - np.indices((256,256))[1])**2)
    homogeneity = np.sum(glcm / (1 + np.abs(np.indices((256,256))[0] - np.indices((256,256))[1])))
    energy = np.sum(glcm**2)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return contrast, homogeneity, energy, entropy


# ==============================
# LBP MANUAL
# ==============================

def lbp_manual(gray):
    lbp = np.zeros_like(gray)
    h, w = gray.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            binary = ''

            neighbors = [
                gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                gray[i+1, j-1], gray[i, j-1]
            ]

            for n in neighbors:
                binary += '1' if n >= center else '0'

            lbp[i, j] = int(binary, 2)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    uniformity = np.sum((hist / np.sum(hist)) ** 2)

    return uniformity


# ==============================
# MAIN PROCESSING
# ==============================

output_file = output_csv + "fitur_buah.csv"

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "nama_file", "kelas",
        "avg_hue", "avg_sat", "avg_val",
        "dark_area_percent",
        "glcm_contrast", "glcm_homogeneity",
        "glcm_energy", "glcm_entropy",
        "lbp_uniformity"
    ])

    for kelas, path in input_folders.items():
        images = glob.glob(path + "*.jpg")

        for img_path in images:
            img = cv2.imread(img_path)

            if img is None:
                continue

            # PREPROCESS
            img = cv2.resize(img, (300, 300))
            blur = cv2.GaussianBlur(img, (5, 5), 0)

            # HSV
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            # RATA-RATA WARNA
            avg_hue = np.mean(hsv[:,:,0])
            avg_sat = np.mean(hsv[:,:,1])
            avg_val = np.mean(hsv[:,:,2])

            # MASK AREA GELAP (busuk)
            dark_mask = cv2.inRange(hsv[:,:,2], 0, 50)
            dark_percent = (np.sum(dark_mask > 0) / dark_mask.size) * 100

            # SIMPAN MASK
            mask_filename = output_mask + os.path.basename(img_path).replace(".jpg", "_mask.png")
            cv2.imwrite(mask_filename, dark_mask)

            # GLCM
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            contrast, homogeneity, energy, entropy = glcm_manual(gray)

            # LBP
            uniformity = lbp_manual(gray)

            # SIMPAN FITUR
            writer.writerow([
                os.path.basename(img_path), kelas,
                avg_hue, avg_sat, avg_val,
                dark_percent,
                contrast, homogeneity, energy, entropy,
                uniformity
            ])

print("\nSELESAI! 🎉")
print("Hasil CSV disimpan di:", output_file)
print("Mask disimpan di folder output/mask/")
