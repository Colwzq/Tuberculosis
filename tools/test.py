# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.apis.test import single_gpu_test_cls
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--txt", help="output class result")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--cls-filter",
        type=bool,
        default=False,
        help="filter the bbox with cls(default: False)",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both "
            "specified, --options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    # ------------------ Debug 专用参数设置 ------------------
    # 1. 让 parse_args 解析空列表，忽略实际命令行参数
    args = parse_args()

    # 2. 基础文件路径
    args.config = "configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py"
    args.checkpoint = "work_dirs/symformer_retinanet_p2t_cls/latest.pth"
    args.out = "work_dirs/symformer_retinanet_p2t_cls/result/result.pkl"
    args.txt = "work_dirs/symformer_retinanet_p2t_cls/result/cls_result.txt"

    # 3. 布尔开关 (Action="store_true")
    # 注意：命令行中的中划线 "-" 在 Python 变量里必须变成下划线 "_"
    args.format_only = True  # 对应 --format-only
    args.cls_filter = True  # 对应 --cls-filter

    # 4. 字典参数 (DictAction)
    # 命令行里的 --options "key=value" 需要转换成 Python 字典
    # 代码逻辑里 options 会被赋给 eval_options，所以直接设置 eval_options
    args.eval_options = {
        "jsonfile_prefix=work_dirs/symformer_retinanet_p2t_cls/result/bbox_result"
    }
    args.options = None  # 防止冲突，置空即可

    # 5. 其他必要参数 (根据脚本可能需要的默认值)
    args.show = False
    args.gpu_collect = False
    args.tmpdir = None
    args.cfg_options = None
    args.launcher = "none"
    args.local_rank = 0

    # 环境变量设置 (模拟 CUDA_VISIBLE_DEVICES=0)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ------------------------------------------------------

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json_file = osp.join(args.work_dir, f"eval_{timestamp}.json")

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not args.cls_filter:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(
                model, data_loader, args.show, args.show_dir, args.show_score_thr
            )
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    else:
        if distributed:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
        else:  # only use single gpu
            model = MMDataParallel(model, device_ids=[0])
            outputs, class_preds = single_gpu_test_cls(
                model, data_loader, args.show, args.show_dir, args.show_score_thr
            )
            # use class_preds filter the bbox
            # TO CHECK
            import torch.nn as nn

            softmax = nn.Softmax(dim=1)
            class_preds = softmax(torch.cat(class_preds).view(-1, 3)).cuda()
            v, max_idx = class_preds.max(dim=1)
            new_outputs = []
            for idx, output in enumerate(outputs):
                if max_idx[idx] == 2:
                    new_outputs.append(output)
                else:
                    import numpy as np

                    output = [
                        np.zeros((0, 5), dtype=np.float32),
                        np.zeros((0, 5), dtype=np.float32),
                    ]
                    new_outputs.append(output)
            if cfg.get("finetune_state", None) is not None:
                print("classfication filtering!")
                outputs = new_outputs
            outputs = new_outputs

            v, _ = class_preds.max(dim=1)
            class_preds = class_preds.cpu().numpy()
            import numpy as np

            # save class_presds
            if args.txt:
                os.makedirs(os.path.dirname(args.txt), exist_ok=True)
                f = open(args.txt, "w")
                # 确保预测数量和数据集样本数量一致
                assert len(class_preds) == len(
                    dataset
                ), f"Preds len {len(class_preds)} != Dataset len {len(dataset)}"

                for i, prob in enumerate(class_preds):
                    # 从 dataset 中获取对应的图片信息
                    # dataset.data_infos 存储了所有样本的元数据
                    img_info = dataset.data_infos[i]
                    # 获取文件名
                    file_name = img_info.get("file_name", f"unknown_{i}")

                    # 组合输出：文件名 + 空格 + 概率
                    # 例如: 001.jpg 0.1 0.9 0.0
                    prob_str = " ".join([str(j) for j in prob])
                    f.writelines(f"{file_name} {prob_str}\n")
                # for i in class_preds:
                #     f.writelines(" ".join([str(j) for j in i]))
                #     f.writelines("\n")
                f.close()

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        # =================== DEBUG START ===================
        print("\n" + "=" * 30)
        print("Debugging IndexError:")

        # 1. 打印数据集认为它有多少类
        print(f"[Dataset Info]")
        print(f"Dataset class names: {dataset.CLASSES}")
        print(f"Len(dataset.CLASSES): {len(dataset.CLASSES)}")
        # coco数据集内部有一个cat_ids映射，报错就是这里越界
        if hasattr(dataset, "cat_ids"):
            print(f"Len(dataset.cat_ids): {len(dataset.cat_ids)}")

        # 2. 打印模型实际输出了多少个类别的结果（检查前几个样本）
        print(f"\n[Model Output Info]")
        if len(outputs) > 0:
            # 检查第一个样本
            print(f"Sample 0 output length: {len(outputs[0])}")

            # 检查所有样本中是否存在异常长度
            lengths = [len(o) for o in outputs]
            max_len = max(lengths)
            min_len = min(lengths)
            print(f"Max output length in all samples: {max_len}")
            print(f"Min output length in all samples: {min_len}")

            if max_len > len(dataset.CLASSES):
                print(f"!!! CRITICAL MISMATCH !!!")
                print(
                    f"Model outputs {max_len} classes, but dataset only expects {len(dataset.CLASSES)}."
                )
                print(
                    f"This is why 'cat_ids[label]' fails when label >= {len(dataset.CLASSES)}."
                )
        else:
            print("Outputs is empty!")
        print("=" * 30 + "\n")
        # =================== DEBUG END ===================
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == "__main__":
    main()
