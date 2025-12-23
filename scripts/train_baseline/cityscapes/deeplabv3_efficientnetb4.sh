python train_baseline.py \
    --model deeplabv3_efficientnet \
    --backbone efficientnet_b4 \
    --dataset citys \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/baseline/cityscapes/ \
    --save-dir-name deeplabv3_efficientnetb4_citys_baseline \
    --aspp_out_channels 512 \
    --pretrained-base /home/sergey/DEV/AI/AEI/pretrained_backbones/efficientnet_b4.pth       #from  pytorch image models
    
    
