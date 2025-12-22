CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 1366 \
    train_baseline.py \
    --model deeplab_mobile \
    --backbone mobilevit_xx_small \
    --dataset citys \
    --batch-size 8 \
    --lr 0.0009 \
    --max-iterations 80000 \
    --optimizer-type adamw \
    --weight-decay 0.01 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/baseline/cityscapes/ \
    --save-dir-name deeplab_mobile_mobilevit_xx_small_citys_baseline \
    --pretrained-base /home/sergey/DEV/AI/AEI/pretrained_backbones//mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth
