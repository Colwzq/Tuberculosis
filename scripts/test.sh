CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py \
    configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_jailcxr.py \
    work_dirs/symformer_retinanet_p2t_cls/latest.pth \
    --out work_dirs/symformer_retinanet_p2t_cls_jail/result/result.pkl \
    --format-only --cls-filter True \
    --options "jsonfile_prefix=work_dirs/symformer_retinanet_p2t_cls_jail/result/bbox_result" \
    --txt work_dirs/symformer_retinanet_p2t_cls_jail/result/cls_result.txt

# CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py \
#     configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py \
#     work_dirs/symformer_retinanet_p2t_cls/latest.pth \
#     --out work_dirs/symformer_retinanet_p2t_cls/result/result.pkl \
#     --format-only --cls-filter True \
#     --options "jsonfile_prefix=work_dirs/symformer_retinanet_p2t_cls/result/bbox_result" \
#     --txt work_dirs/symformer_retinanet_p2t_cls/result/cls_result.txt