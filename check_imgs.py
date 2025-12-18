import os
import cv2
from pathlib import Path
from tqdm import tqdm

# 根据你的 config 文件，图片路径在这里
DATA_ROOT = "/home/colwzq/data/medical_datasets/TBX11K/imgs/"


def check_images(root_dir):
    print(f"正在扫描目录: {root_dir}")
    # 支持常见的图片格式
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    corrupt_files = []

    # 遍历所有文件
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in exts:
                image_files.append(os.path.join(root, file))

    print(f"共发现 {len(image_files)} 张图片，开始检查完整性...")

    for img_path in tqdm(image_files):
        try:
            # 尝试读取图片
            img = cv2.imread(img_path)
            # 如果读取结果为 None，说明解码失败
            if img is None:
                print(f"\n[发现损坏] 无法读取: {img_path}")
                corrupt_files.append(img_path)
        except Exception as e:
            print(f"\n[发生异常] {img_path}: {e}")
            corrupt_files.append(img_path)

    print("-" * 30)
    if corrupt_files:
        print(f"扫描结束。共发现 {len(corrupt_files)} 个损坏文件:")
        for f in corrupt_files:
            print(f)
        print("\n建议：删除这些图片，并从 annotation json 文件中移除对应的标注信息。")
    else:
        print("扫描结束，未发现明显损坏的图片。")


if __name__ == "__main__":
    check_images(DATA_ROOT)
