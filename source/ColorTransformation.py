"""
Biến đổi màu (Color transformation) - cài đặt thủ công bằng NumPy.
OpenCV chỉ dùng để đọc/ghi ảnh và làm đáp án đối chiếu.

Chạy:  python source/ColorTransformation.py <đường_dẫn_ảnh>
"""
import os
import sys

import cv2
import numpy as np


def clip_u8(x):
    """Làm tròn rồi ép về [0, 255] kiểu uint8."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def bgr_to_gray(img):
    """Gray = 0.299 R + 0.587 G + 0.114 B (OpenCV đọc ảnh theo thứ tự B, G, R)."""
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    return clip_u8(0.114 * b + 0.587 * g + 0.299 * r)


def bgr_to_hsv(img):
    """BGR -> HSV theo quy ước 8-bit của OpenCV: H in [0,179], S và V in [0,255]."""
    x = img.astype(np.float32) / 255.0
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=2)
    d = v - x.min(axis=2)
    s = np.where(v > 0, d / np.maximum(v, 1e-12), 0)

    h = np.zeros_like(v)
    nz = d > 0
    m_r = nz & (v == r)
    m_g = nz & (v == g) & ~m_r
    m_b = nz & ~m_r & ~m_g
    h[m_r] = 60 * (g - b)[m_r] / d[m_r]
    h[m_g] = 120 + 60 * (b - r)[m_g] / d[m_g]
    h[m_b] = 240 + 60 * (r - g)[m_b] / d[m_b]
    h[h < 0] += 360

    h = clip_u8(h / 2).astype(np.int32) % 180  # 360 độ làm tròn thành 180 -> quay về 0
    return np.stack([h.astype(np.uint8), clip_u8(s * 255), clip_u8(v * 255)], axis=2)


def adjust_brightness_contrast(img, alpha=1.2, beta=30):
    """out = alpha * in + beta (alpha: độ tương phản, beta: độ sáng)."""
    return clip_u8(alpha * img.astype(np.float32) + beta)


def negative(img):
    """Ảnh âm bản: 255 - pixel."""
    return 255 - img


def gamma_correction(img, gamma=0.5):
    """out = 255 * (in/255)^gamma. gamma < 1 làm sáng, gamma > 1 làm tối."""
    lut = clip_u8(255.0 * (np.arange(256) / 255.0) ** gamma)
    return lut[img]


def sepia(img):
    """Hiệu ứng ảnh cũ: nhân ma trận 3x3 lên từng pixel (thứ tự BGR)."""
    m = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]], dtype=np.float32)
    return clip_u8(img.astype(np.float32) @ m.T)


def _report(name, mine, ref):
    diff = np.abs(mine.astype(np.float32) - ref.astype(np.float32))
    print(f"{name:<22} max|diff| = {diff.max():5.2f}   mean|diff| = {diff.mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Dùng: python source/ColorTransformation.py <đường_dẫn_ảnh>")

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {sys.argv[1]}")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # So sánh với OpenCV
    _report("gray", bgr_to_gray(img), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    _report("hsv", bgr_to_hsv(img), cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    _report("brightness/contrast", adjust_brightness_contrast(img, 1.2, 30),
            cv2.convertScaleAbs(img, alpha=1.2, beta=30))
    _report("negative", negative(img), cv2.bitwise_not(img))

    # Lưu kết quả
    results = {
        "gray": bgr_to_gray(img),
        "hsv": bgr_to_hsv(img),
        "brightness_contrast": adjust_brightness_contrast(img),
        "negative": negative(img),
        "gamma": gamma_correction(img, 0.5),
        "sepia": sepia(img),
    }
    for name, res in results.items():
        cv2.imwrite(os.path.join(out_dir, f"color_{name}.png"), res)
    print(f"Đã lưu kết quả vào '{out_dir}/'")