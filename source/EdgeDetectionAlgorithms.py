"""
Phát hiện biên cạnh (Edge detection) - cài đặt thủ công bằng NumPy.
OpenCV chỉ dùng để đọc/ghi ảnh và làm đáp án đối chiếu.

Ý tưởng chung: biên = nơi độ sáng thay đổi mạnh = đạo hàm (gradient) lớn.

Chạy:  python source/EdgeDetectionAlgorithms.py <đường_dẫn_ảnh>
"""
import os
import sys

import cv2
import numpy as np

# Dùng lại tích chập và kernel Gauss đã viết ở task làm trơn ảnh
from ImageSmoothingAlgorithms import clip_u8, filter2d, gaussian_kernel

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], np.float32)
PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], np.float32)
LAPLACE = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], np.float32)


def to_gray(img):
    """Ảnh màu BGR -> xám (nếu đã là ảnh xám thì giữ nguyên)."""
    if img.ndim == 2:
        return img
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    return clip_u8(0.114 * b + 0.587 * g + 0.299 * r)


def gradients(gray, kx):
    """Trả về (gx, gy) dạng float. Kernel theo trục y là chuyển vị của kernel theo trục x."""
    return filter2d(gray, kx), filter2d(gray, kx.T)


def sobel_edges(gray):
    """Độ lớn gradient xấp xỉ bằng |gx| + |gy|."""
    gx, gy = gradients(gray, SOBEL_X)
    return clip_u8(np.abs(gx) + np.abs(gy))


def prewitt_edges(gray):
    """Giống Sobel nhưng các hàng/cột có trọng số bằng nhau (không nhấn mạnh hàng giữa)."""
    gx, gy = gradients(gray, PREWITT_X)
    return clip_u8(np.abs(gx) + np.abs(gy))


def laplacian_edges(gray):
    """Đạo hàm bậc 2: biên nằm ở chỗ giá trị đổi dấu. Nhạy với nhiễu hơn Sobel."""
    return clip_u8(np.abs(filter2d(gray, LAPLACE)))


def canny(gray, low=50, high=150, ksize=5, sigma=1.4):
    """Canny gồm 4 bước:
    1. Làm trơn bằng Gauss (giảm nhiễu)
    2. Tính gradient (độ lớn + hướng)
    3. Non-maximum suppression: chỉ giữ pixel lớn nhất theo hướng gradient -> biên mỏng 1 pixel
    4. Ngưỡng kép + hysteresis: biên mạnh giữ luôn, biên yếu chỉ giữ nếu nối với biên mạnh
    """
    # Bước 1-2
    blur = filter2d(gray, gaussian_kernel(ksize, sigma))
    gx, gy = gradients(blur, SOBEL_X)
    mag = np.abs(gx) + np.abs(gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180

    # Bước 3: so pixel với 2 láng giềng nằm dọc theo hướng gradient (4 hướng)
    h, w = mag.shape
    mp = np.pad(mag, 1)

    def nb(dy, dx):  # láng giềng lệch (dy, dx)
        return mp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    d135 = ~(d0 | d45 | d90)
    n1 = np.select([d0, d45, d90, d135], [nb(0, -1), nb(-1, -1), nb(-1, 0), nb(-1, 1)])
    n2 = np.select([d0, d45, d90, d135], [nb(0, 1), nb(1, 1), nb(1, 0), nb(1, -1)])
    nms = np.where((mag >= n1) & (mag >= n2), mag, 0)

    # Bước 4: biên yếu "lan" từ biên mạnh cho đến khi không lan thêm được nữa
    strong = nms > high
    weak = (nms > low) & ~strong
    while True:
        sp = np.pad(strong, 1)
        near = np.zeros_like(strong)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= sp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        grow = weak & near
        if not grow.any():
            break
        strong |= grow
        weak &= ~grow
    return (strong * 255).astype(np.uint8)


def _report(name, mine, ref):
    diff = np.abs(mine.astype(np.float32) - ref.astype(np.float32))
    print(f"{name:<22} max|diff| = {diff.max():5.2f}   mean|diff| = {diff.mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Dùng: python source/EdgeDetectionAlgorithms.py <đường_dẫn_ảnh>")

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {sys.argv[1]}")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    gray = to_gray(img)

    # So sánh với OpenCV (dạng float để thấy sai khác thật, chưa bị cắt về 0-255)
    gx, _ = gradients(gray, SOBEL_X)
    _report("sobel gx", gx, cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    _report("laplacian", filter2d(gray, LAPLACE), cv2.Laplacian(gray, cv2.CV_32F, ksize=1))

    my_canny = canny(gray, 50, 150)
    blurred = clip_u8(filter2d(gray, gaussian_kernel(5, 1.4)))
    cv_canny = cv2.Canny(blurred, 50, 150)  # cv2.Canny không tự làm trơn nên đưa ảnh đã làm trơn vào
    print(f"{'canny':<22} số pixel biên khác nhau = {(my_canny != cv_canny).mean() * 100:.3f}%")

    # Lưu kết quả
    results = {
        "sobel": sobel_edges(gray),
        "prewitt": prewitt_edges(gray),
        "laplacian": laplacian_edges(gray),
        "canny": my_canny,
        "canny_opencv": cv_canny,
    }
    for name, res in results.items():
        cv2.imwrite(os.path.join(out_dir, f"edge_{name}.png"), res)
    print(f"Đã lưu kết quả vào '{out_dir}/'")