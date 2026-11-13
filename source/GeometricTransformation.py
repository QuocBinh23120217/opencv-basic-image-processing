"""
Biến đổi hình học (Geometric transformation) - cài đặt thủ công bằng NumPy.
OpenCV chỉ dùng để đọc/ghi ảnh và làm đáp án đối chiếu.

Ý tưởng chung (inverse mapping): với mỗi pixel của ảnh ĐÍCH, tính ngược về
toạ độ trên ảnh NGUỒN rồi nội suy -> kết quả không bị "lỗ hổng".

Chạy:  python source/GeometricTransformation.py <đường_dẫn_ảnh>
"""
import os
import sys

import cv2
import numpy as np


def clip_u8(x):
    """Làm tròn rồi ép về [0, 255] kiểu uint8."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def bilinear_sample(img, xs, ys):
    """Lấy giá trị ảnh tại toạ độ thực (xs, ys) bằng nội suy song tuyến tính.
    Điểm nằm ngoài ảnh -> 0 (đen)."""
    h, w = img.shape[:2]
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    wx, wy = xs - x0, ys - y0  # phần lẻ = trọng số nội suy
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)

    if img.ndim == 3:
        wx, wy, valid = wx[..., None], wy[..., None], valid[..., None]
    img = img.astype(np.float32)
    top = img[y0c, x0c] * (1 - wx) + img[y0c, x1c] * wx
    bot = img[y1c, x0c] * (1 - wx) + img[y1c, x1c] * wx
    out = top * (1 - wy) + bot * wy
    return clip_u8(np.where(valid, out, 0))


def warp_affine(img, M, out_size=None):
    """Áp dụng phép biến đổi affine M (ma trận 2x3, chiều thuận: nguồn -> đích)."""
    h, w = img.shape[:2]
    ow, oh = out_size if out_size else (w, h)
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))  # đảo ngược: đích -> nguồn
    X, Y = np.meshgrid(np.arange(ow, dtype=np.float32),
                       np.arange(oh, dtype=np.float32))
    xs = inv[0, 0] * X + inv[0, 1] * Y + inv[0, 2]
    ys = inv[1, 0] * X + inv[1, 1] * Y + inv[1, 2]
    return bilinear_sample(img, xs, ys)


def translation_matrix(tx, ty):
    """Tịnh tiến: dịch sang phải tx pixel, xuống dưới ty pixel."""
    return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)


def rotation_matrix(center, angle_deg, scale=1.0):
    """Quay quanh `center`. Giống cv2.getRotationMatrix2D: góc dương = ngược chiều kim đồng hồ."""
    cx, cy = center
    t = np.deg2rad(angle_deg)
    a, b = scale * np.cos(t), scale * np.sin(t)
    return np.array([[a, b, (1 - a) * cx - b * cy],
                     [-b, a, b * cx + (1 - a) * cy]], dtype=np.float32)


def translate(img, tx, ty):
    return warp_affine(img, translation_matrix(tx, ty))


def rotate(img, angle_deg, scale=1.0):
    h, w = img.shape[:2]
    return warp_affine(img, rotation_matrix((w / 2, h / 2), angle_deg, scale))


def resize_bilinear(img, fx, fy):
    """Phóng to / thu nhỏ. Dùng quy ước tâm pixel như OpenCV: src = (dst + 0.5) / s - 0.5."""
    h, w = img.shape[:2]
    ow, oh = int(round(w * fx)), int(round(h * fy))
    X, Y = np.meshgrid(np.arange(ow, dtype=np.float32),
                       np.arange(oh, dtype=np.float32))
    xs = np.clip((X + 0.5) / fx - 0.5, 0, w - 1)
    ys = np.clip((Y + 0.5) / fy - 0.5, 0, h - 1)
    return bilinear_sample(img, xs, ys)


def flip(img, mode):
    """mode: 'h' lật ngang, 'v' lật dọc."""
    return img[:, ::-1] if mode == "h" else img[::-1]


def _report(name, mine, ref):
    diff = np.abs(mine.astype(np.float32) - ref.astype(np.float32))
    print(f"{name:<22} max|diff| = {diff.max():5.2f}   mean|diff| = {diff.mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Dùng: python source/GeometricTransformation.py <đường_dẫn_ảnh>")

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {sys.argv[1]}")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    h, w = img.shape[:2]

    # So sánh với OpenCV
    M_t = translation_matrix(50, 30)
    _report("translation", translate(img, 50, 30), cv2.warpAffine(img, M_t, (w, h)))
    M_r = rotation_matrix((w / 2, h / 2), 30)
    _report("rotation 30 deg", rotate(img, 30), cv2.warpAffine(img, M_r, (w, h)))
    _report("resize x1.5", resize_bilinear(img, 1.5, 1.5),
            cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR))
    _report("resize x0.5", resize_bilinear(img, 0.5, 0.5),
            cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
    _report("flip horizontal", flip(img, "h"), cv2.flip(img, 1))

    # Lưu kết quả
    results = {
        "translate": translate(img, 50, 30),
        "rotate": rotate(img, 30),
        "up": resize_bilinear(img, 1.5, 1.5),
        "down": resize_bilinear(img, 0.5, 0.5),
        "flip_h": flip(img, "h"),
        "flip_v": flip(img, "v"),
    }
    for name, res in results.items():
        cv2.imwrite(os.path.join(out_dir, f"geo_{name}.png"), res)
    print(f"Đã lưu kết quả vào '{out_dir}/'")