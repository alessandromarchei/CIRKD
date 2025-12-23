python eval.py \
    --model psp_efficientnet \
    --backbone efficientnet_b0 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --pretrained checkpoints/baseline/cityscapes/psp_efficientnetb0_2_citys_baseline/psp_efficientnet_efficientnet_b0_citys_best_model.pth


#called the training setup with LR = 0.001 (instead of the 0.01 used in the baseline)