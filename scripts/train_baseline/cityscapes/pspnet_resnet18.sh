python train_baseline.py \
    --model psp \
    --backbone resnet18 \
    --dataset citys \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/baseline/cityscapes/ \
    --save-dir-name psp_resnet18_citys_baseline \
    --pretrained-base /home/sergey/DEV/AI/AEI/pretrained_backbones/resnet18.pth

