python train_baseline.py \
    --model psp_mobile \
    --backbone mobilenetv3_small \
    --dataset citys \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/cirkd_checkpoints/cityscapes/ \
    --save-dir-name psp_mobile_mobilenetv3_small_citys_baseline \
    --pretrained-base /home/sergey/DEV/AI/AEI/pretrained_backbones/mobilenet_v3_small.pth

