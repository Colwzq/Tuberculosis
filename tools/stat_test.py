import json
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    recall_score,
)

# ================= 配置路径 =================
# 1. 你生成的测试集 JSON 路径 (为了获取文件名和顺序)
JSON_PATH = "/home/colwzq/data/jail_cxr/test_coco_format.json"

# 2. 你的推理结果文件 (纯概率，一行三个数)
RESULT_TXT_PATH = "work_dirs/symformer_retinanet_p2t_cls_jail/result/cls_result.txt"
# ===========================================


def get_gt_label(file_name):
    """
    【重要】请根据你的实际文件名规则修改这里！
    我们需要根据文件名返回真实标签：
    0: Healthy
    1: Sick non-TB
    2: TB
    """
    name = file_name.lower()

    # 示例逻辑：根据文件名包含的字符串判断
    # 如果你的文件名是 '2023_cxr_tb_001.png' 这种格式
    if "tb" in name:
        return 2  # TB
    elif "sick" in name or "_s_" in name:
        return 1  # Sick non-TB
    else:
        return 0  # Healthy (默认)


def main():
    # 1. 加载 Ground Truth (从 JSON 读取顺序)
    print(f"正在加载 JSON: {JSON_PATH} ...")
    with open(JSON_PATH, "r") as f:
        coco_data = json.load(f)

    img_list = coco_data["images"]
    y_true = []
    filenames = []

    for img_info in img_list:
        fname = img_info["file_name"]
        label = get_gt_label(fname)
        y_true.append(label)
        filenames.append(fname)

    y_true = np.array(y_true)

    # 2. 加载 预测结果
    print(f"正在加载 Result: {RESULT_TXT_PATH} ...")
    try:
        # 读取纯数字文本
        y_scores = np.loadtxt(RESULT_TXT_PATH)
    except Exception as e:
        print(f"读取 result.txt 失败: {e}")
        return

    # 3. 校验数量是否对齐
    if len(y_true) != len(y_scores):
        print(f"错误！数据长度不一致！")
        print(f"JSON 图片数: {len(y_true)}")
        print(f"Result 行数: {len(y_scores)}")
        print("请检查 result.txt 是否是完整对应这个 json 生成的。")
        return

    print(f"数据校验通过，共 {len(y_true)} 条数据。")

    # =========================================================
    # 指标计算 (重点关注 TB 类，即 Class Index 2)
    # =========================================================

    # 设定 TB 为正类 (Positive)，其他为负类
    # 真实标签二值化: 1 if TB, else 0
    y_true_binary = (y_true == 2).astype(int)

    # 预测概率: 取出第 3 列 (Index 2) 作为 TB 的概率
    y_score_tb = y_scores[:, 2]

    # 预测类别: 选概率最大的那个类
    y_pred_cls = np.argmax(y_scores, axis=1)
    # 预测二值化: 如果最大概率类别是 2，则预测为 TB
    y_pred_binary = (y_pred_cls == 2).astype(int)

    # 1. Accuracy (整体分类准确率，或者二分类准确率)
    # 这里计算二分类准确率 (TB vs Non-TB)
    acc = accuracy_score(y_true_binary, y_pred_binary)

    # 2. AUC (Area Under ROC Curve) - 衡量 TB 排序能力
    auc = roc_auc_score(y_true_binary, y_score_tb)

    # 3. 混淆矩阵 & Sensitivity / Specificity
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    # Sensitivity (Sensitivity = Recall of Positive Class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 4. Ave. Prec (Classification AP)
    # 这对应 PR 曲线下的面积 (AUPR)
    ap = average_precision_score(y_true_binary, y_score_tb)

    # 5. Ave. Rec (Classification AR)
    # 在分类任务中，AR 通常等同于 Sensitivity (Recall)
    # 或者有时指多分类的 Macro-Recall。针对 TB 筛查，看 Sensitivity 即可。
    ar = recall_score(y_true_binary, y_pred_binary)

    # ================= 打印结果 =================
    print("\n" + "=" * 40)
    print("      TB 分类/诊断 评估结果      ")
    print("=" * 40)
    print(f"Total Samples: {len(y_true)}")
    print(f"GT Distribution: Healthy/Sick={np.sum(y_true!=2)}, TB={np.sum(y_true==2)}")
    print("-" * 40)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<20} | {acc:.4f}")
    print(f"{'AUC (TB)':<20} | {auc:.4f}")
    print(f"{'Sensitivity':<20} | {sensitivity:.4f}")
    print(f"{'Specificity':<20} | {specificity:.4f}")
    print(f"{'Ave. Prec (AP)':<20} | {ap:.4f}")
    print(f"{'Ave. Rec  (AR)':<20} | {ar:.4f}")
    print("-" * 40)
    print("Confusion Matrix (TB vs Non-TB):")
    print(f"TN: {tn}\t FP: {fp}")
    print(f"FN: {fn}\t TP: {tp}")
    print("=" * 40)


if __name__ == "__main__":
    main()
