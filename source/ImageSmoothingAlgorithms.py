"""
Làm trơn ảnh (Image smoothing) - cài đặt thủ công bằng NumPy.
OpenCV chỉ dùng để đọc/ghi ảnh và làm đáp án đối chiếu.

Ý tưởng chung: mỗi pixel mới được tính từ các pixel lân cận trong một
cửa sổ nhỏ -> các biến động đột ngột (nhiễu) bị "trung hoà" bớt.

Chạy:  python source/ImageSmoothingAlgorithms.py <đường_dẫn_ảnh>
"""
import os
import sys

import cv2
import numpy as np


def clip_u8(x):
    """Làm tròn rồi ép về [0, 255] kiểu uint8."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def filter2d(img, kernel):
    """Tương quan 2D (giống cv2.filter2D), biên phản xạ (reflect 101).

    Chỉ lặp trên kernel (nhỏ); phần ảnh được tính bằng vector hoá
    nên nhanh hơn nhiều so với duyệt từng pixel bằng 2 vòng for.
    """
    k = np.asarray(kernel, dtype=np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    x = img.astype(np.float32)
    pad = ((ph, ph), (pw, pw)) + ((0, 0),) * (x.ndim - 2)
    xp = np.pad(x, pad, mode="reflect")
    h, w = x.shape[:2]
    out = np.zeros_like(x)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * xp[i:i + h, j:j + w]
    return out


def box_blur(img, ksize=5):
    """Lọc trung bình: mọi pixel trong cửa sổ có trọng số bằng nhau."""
    k = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return clip_u8(filter2d(img, k))


# OpenCV dùng sẵn bảng kernel 1D cho ksize nhỏ khi sigma <= 0
_SMALL_GAUSS = {
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
}


def gaussian_kernel(ksize, sigma):
    """Kernel Gauss 2D: pixel càng gần tâm càng có trọng số lớn."""
    if sigma <= 0:
        if ksize in _SMALL_GAUSS:
            g = np.array(_SMALL_GAUSS[ksize], np.float32)
            return np.outer(g, g)
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # cách OpenCV chọn sigma mặc định
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)  # kernel 2D = tích ngoài của 2 kernel 1D
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img, ksize=5, sigma=0):
    return clip_u8(filter2d(img, gaussian_kernel(ksize, sigma)))


def median_blur(img, ksize=5):
    """Lọc trung vị (phi tuyến): lấy giá trị đứng giữa của cửa sổ.
    Rất hiệu quả với nhiễu muối tiêu vì pixel nhiễu luôn nằm ở 2 đầu dãy đã sắp xếp."""
    p = ksize // 2
    pad = ((p, p), (p, p)) + ((0, 0),) * (img.ndim - 2)
    xp = np.pad(img, pad, mode="edge")  # medianBlur của OpenCV nhân bản pixel biên
    h, w = img.shape[:2]
    stack = np.stack([xp[i:i + h, j:j + w]
                      for i in range(ksize) for j in range(ksize)], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def add_salt_pepper(img, amount=0.05, seed=0):
    """Thêm nhiễu muối tiêu để thấy rõ tác dụng của từng bộ lọc."""
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    r = rng.random(img.shape[:2])
    noisy[r < amount / 2] = 0          # tiêu (đen)
    noisy[r > 1 - amount / 2] = 255    # muối (trắng)
    return noisy


def _report(name, mine, ref):
    diff = np.abs(mine.astype(np.float32) - ref.astype(np.float32))
    print(f"{name:<22} max|diff| = {diff.max():5.2f}   mean|diff| = {diff.mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Dùng: python source/ImageSmoothingAlgorithms.py <đường_dẫn_ảnh>")

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {sys.argv[1]}")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # So sánh với OpenCV
    _report("box 5x5", box_blur(img, 5), cv2.blur(img, (5, 5)))
    _report("gaussian 5x5", gaussian_blur(img, 5), cv2.GaussianBlur(img, (5, 5), 0))
    _report("median 5x5", median_blur(img, 5), cv2.medianBlur(img, 5))

    # Lưu kết quả: ảnh gốc làm trơn, và ảnh nhiễu muối tiêu được khử nhiễu
    noisy = add_salt_pepper(img)
    results = {
        "box": box_blur(img, 5),
        "gaussian": gaussian_blur(img, 5),
        "median": median_blur(img, 5),
        "noisy": noisy,
        "noisy_box": box_blur(noisy, 5),
        "noisy_gaussian": gaussian_blur(noisy, 5),
        "noisy_median": median_blur(noisy, 5),
    }
    for name, res in results.items():
        cv2.imwrite(os.path.join(out_dir, f"smooth_{name}.png"), res)
    print(f"Đã lưu kết quả vào '{out_dir}/'")