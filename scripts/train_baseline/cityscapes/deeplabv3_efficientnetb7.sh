python train_baseline.py \
    --model deeplabv3_efficientnet \
    --backbone efficientnet_b7 \
    --dataset citys \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/baseline/cityscapes/ \
    --save-dir-name deeplabv3_efficientnetb7_4_citys_baseline \
    --aspp_out_channels 512 \
    --out_indices 3 \
    --pretrained-base /home/sergey/DEV/AI/AEI/pretrained_backbones/efficientnet_b7.pth       #from  pytorch image models
    
    
