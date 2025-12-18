# # step I: train detection
# CUDA_VISIBLE_DEVICE=0 python tools/train.py \
#     --config configs/symformer/symformer_retinanet_p2t_fpn_2x_TBX11K.py \
#     --work-dir work_dirs/symformer_retinanet_p2t/ \
#     --no-validate

# step II: train classification
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --config configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py \
    --work-dir work_dirs/symformer_retinanet_p2t_cls/ \
    --no-validate