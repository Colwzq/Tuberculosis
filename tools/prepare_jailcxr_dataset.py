import os
import json
import glob
import numpy as np
import pydicom
import cv2
from tqdm import tqdm
from datetime import datetime

# ================= 配置路径 =================
# 原始数据所在的根目录
SRC_ROOT = "/home/colwzq/data/jail_cxr/"
# 包含数据的子文件夹
SUB_DIRS = ["2023_cxr", "2024_cxr", "2025_cxr"]

# 目标图片保存目录 (将png存放在这里)
DST_IMG_DIR = os.path.join(SRC_ROOT, "imgs")

# 目标 JSON 文件保存路径
JSON_SAVE_PATH = os.path.join(SRC_ROOT, "jail_cxr_test.json")

# 作者使用了 512x512，我们也统一到这个尺寸方便计算
# 如果不想缩放，设为 None
TARGET_SIZE = (512, 512)

# ===========================================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def robust_normalize(pixel_array):
    """
    鲁棒归一化：解决极值噪点导致肺部过暗的问题 (坑点1 & 2)
    """
    # # remove watermark
    # h, w = pixel_array.shape
    # # 假设水印在右下角 50x50 区域，根据实际情况调整
    # pixel_array[h - 55 : h, w - 55 : w] = 0
    # remove watermark
    h, w = pixel_array.shape
    pixel_array = pixel_array[: h - 55, : w - 55]

    pixel_array = pixel_array.astype(np.float32)

    # 1. 极值截断 (Percentile Clipping)
    # 剔除 0.5% 最暗和 0.5% 最亮的异常像素（如金属伪影或直接曝光区域）
    lower_bound = np.percentile(pixel_array, 0.5)
    upper_bound = np.percentile(pixel_array, 99.5)

    pixel_array = np.clip(pixel_array, lower_bound, upper_bound)

    # 2. 归一化到 0-255
    if upper_bound == lower_bound:
        return np.zeros_like(pixel_array, dtype=np.uint8)

    normalized = (pixel_array - lower_bound) / (upper_bound - lower_bound) * 255.0

    return normalized.astype(np.uint8)


def normalize_dicom_image(pixel_array):
    """
    将DICOM的原始像素值归一化到 0-255 并转换为 uint8，方便保存为 png
    注意：医学图像可能有不同的位深(12bit, 16bit)，这里采用 Min-Max 归一化。
    """

    if pixel_array.max() == pixel_array.min():
        return np.zeros_like(pixel_array, dtype=np.uint8)

    pixel_array = pixel_array.astype(float)
    # 简单的线性归一化
    normalized = (
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min())
        * 255.0
    )
    return normalized.astype(np.uint8)


def is_chest_dcm(file_path):
    KEYWORDS = ["CHEST", "THORAX", "LUNG"]
    """
    读取DCM文件头，判断是否包含胸部相关的描述
    """
    try:
        # stop_before_pixels=True 可以避免读取图像数据，显著提高速度
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)

        # 需要检查的DICOM标签列表
        tags_to_check = [
            "BodyPartExamined",  # 检查部位
            "StudyDescription",  # 检查描述
            "SeriesDescription",  # 序列描述
            "ProtocolName",  # 协议名称
        ]

        found_keywords = []
        tags_values = {}

        for tag in tags_to_check:
            # 获取标签值，如果不存在则为空字符串
            value = str(ds.get(tag, "")).strip().upper()
            tags_values[tag] = value

            # 检查是否有任何关键词出现在该标签值中
            for kw in KEYWORDS:
                if kw in value:
                    found_keywords.append(f"{tag}:{value}")

        # 如果找到了关键词，返回True
        if found_keywords:
            return (
                True,
                "; ".join(found_keywords),
                tags_values.get("BodyPartExamined", ""),
            )
        else:
            return False, "Not Found", tags_values.get("BodyPartExamined", "")

    except Exception as e:
        return False, f"Error: {str(e)}", "N/A"


def main():
    ensure_dir(DST_IMG_DIR)

    images_info = []
    # 这里的 categories 可以为空，或者写一个 dummy 类别，
    # 如果只是做纯 inference (test)，有时候不需要，但为了兼容性建议加上背景或通用类别
    categories = [
        {"id": 1, "name": "ActiveTuberculosis", "supercategory": "Tuberculosis"},
        {
            "id": 2,
            "name": "ObsoletePulmonaryTuberculosis",
            "supercategory": "Tuberculosis",
        },
        {"id": 3, "name": "PulmonaryTuberculosis", "supercategory": "Tuberculosis"},
    ]

    # 图片 ID 计数器
    img_id = 0

    print(f"开始处理数据，目标目录: {DST_IMG_DIR}")

    for sub_dir in SUB_DIRS:
        full_sub_dir = os.path.join(SRC_ROOT, sub_dir)
        # 搜索所有 dcw 文件
        files = glob.glob(os.path.join(full_sub_dir, "*.dcm"))

        print(f"正在处理文件夹: {sub_dir}, 包含 {len(files)} 个文件")

        for file_path in tqdm(files):
            try:
                # 1. 读取 DCW (DICOM)
                ds = pydicom.dcmread(file_path)
                is_chest = is_chest_dcm(file_path)
                if not is_chest[0]:
                    continue

                # 检查 Photometric Interpretation，有些X光如果是 MONOCHROME1，图像是反色的(骨头是黑的)
                # 通常肺部X光希望骨头/高密度是白的。如果是 MONOCHROME1，通常需要反转
                if (
                    hasattr(ds, "PhotometricInterpretation")
                    and ds.PhotometricInterpretation == "MONOCHROME1"
                ):
                    img_array = 255 - img_array

                # 获取原始宽高
                height = ds.Rows
                width = ds.Columns

                img_array = robust_normalize(ds.pixel_array)

                # 获取像素数据并保存
                if TARGET_SIZE is not None:
                    # 使用插值算法缩放
                    img_to_save = cv2.resize(
                        img_array, TARGET_SIZE, interpolation=cv2.INTER_LINEAR
                    )
                    final_h, final_w = TARGET_SIZE[1], TARGET_SIZE[0]
                else:
                    img_to_save = img_array
                    final_h, final_w = img_array.shape[0], img_array.shape[1]

                # 2. 转换并保存 PNG
                # 生成一个新的文件名，建议带上年份前缀防止重名
                # 例如: 2023_cxr_unknown_3295.png
                original_name = os.path.splitext(os.path.basename(file_path))[0]
                new_filename = f"{sub_dir}_{original_name}.png"
                new_filename = f"{original_name}.png"
                dst_path = os.path.join(DST_IMG_DIR, new_filename)
                cv2.imwrite(dst_path, img_to_save)

                # 3. 构建 JSON 信息
                # MMDetection 通常只需要 file_name, height, width, id
                img_info = {
                    "id": img_id,
                    "file_name": new_filename,  # 注意：这里通常存相对路径，相对于 mmdet config 中的 data_root
                    "width": final_w,
                    "height": final_h,
                    "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "license": 1,
                    "coco_url": "",
                    "flickr_url": "",
                }
                images_info.append(img_info)

                img_id += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # 构建最终的 COCO JSON 结构
    coco_output = {
        "info": {"description": "Jail CXR Test Dataset converted from DCM"},
        "licenses": [],
        "images": images_info,
        "annotations": [],  # 测试集不需要标注，留空
        "categories": categories,
    }

    # 保存 JSON
    with open(JSON_SAVE_PATH, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"处理完成！")
    print(f"图片已保存在: {DST_IMG_DIR}")
    print(f"JSON 已生成在: {JSON_SAVE_PATH}")
    print(f"共处理图片: {len(images_info)} 张")


if __name__ == "__main__":
    main()
