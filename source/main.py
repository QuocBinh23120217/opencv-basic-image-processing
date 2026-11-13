"""
Điểm vào chung: chạy các thao tác xử lý ảnh trên một ảnh và lưu kết quả.
Ảnh đầu vào sẽ lưu vào thư mục project
Chạy với các lệnh sau:
    python source/main.py <đường_dẫn_ảnh>                 # chạy cả 4 phần
    python source/main.py <đường_dẫn_ảnh> --task edge     # chỉ chạy 1 phần
    python source/main.py <đường_dẫn_ảnh> -o ketqua       # đổi thư mục lưu
"""
import argparse
import os

import cv2

import ColorTransformation as color
import EdgeDetectionAlgorithms as edge
import GeometricTransformation as geo
import ImageSmoothingAlgorithms as smooth


def run_color(img):
    return {
        "gray": color.bgr_to_gray(img),
        "hsv": color.bgr_to_hsv(img),
        "brightness_contrast": color.adjust_brightness_contrast(img),
        "negative": color.negative(img),
        "gamma": color.gamma_correction(img, 0.5),
        "sepia": color.sepia(img),
    }


def run_geometric(img):
    return {
        "translate": geo.translate(img, 50, 30),
        "rotate": geo.rotate(img, 30),
        "up": geo.resize_bilinear(img, 1.5, 1.5),
        "down": geo.resize_bilinear(img, 0.5, 0.5),
        "flip_h": geo.flip(img, "h"),
        "flip_v": geo.flip(img, "v"),
    }


def run_smoothing(img):
    noisy = smooth.add_salt_pepper(img)
    return {
        "box": smooth.box_blur(img, 5),
        "gaussian": smooth.gaussian_blur(img, 5),
        "median": smooth.median_blur(img, 5),
        "noisy": noisy,
        "noisy_box": smooth.box_blur(noisy, 5),
        "noisy_gaussian": smooth.gaussian_blur(noisy, 5),
        "noisy_median": smooth.median_blur(noisy, 5),
    }


def run_edge(img):
    gray = edge.to_gray(img)
    return {
        "sobel": edge.sobel_edges(gray),
        "prewitt": edge.prewitt_edges(gray),
        "laplacian": edge.laplacian_edges(gray),
        "canny": edge.canny(gray, 50, 150),
    }


TASKS = {
    "color": run_color,
    "geometric": run_geometric,
    "smoothing": run_smoothing,
    "edge": run_edge,
}


def main():
    ap = argparse.ArgumentParser(description="Các thao tác xử lý ảnh cơ bản (cài đặt thủ công)")
    ap.add_argument("image", help="đường dẫn ảnh đầu vào")
    ap.add_argument("--task", choices=["all", *TASKS], default="all")
    ap.add_argument("-o", "--out", default="output", help="thư mục lưu kết quả")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {args.image}")

    selected = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    for task_name, fn in selected.items():
        task_dir = os.path.join(args.out, task_name)
        os.makedirs(task_dir, exist_ok=True)
        results = fn(img)
        for name, res in results.items():
            cv2.imwrite(os.path.join(task_dir, f"{name}.png"), res)
        print(f"[{task_name}] đã lưu {len(results)} ảnh vào '{task_dir}/'")


if __name__ == "__main__":
    main()