CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12397 \
    train_cirkdv2.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3_mobilenet_ssseg \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --lambda-fitnet 1. \
    --lambda-minibatch-channel 1. \
    --lambda-memory-channel 0.1 \
    --lambda-channel-kd 100. \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/cirkd_v2/cityscapes/ \
    --save-dir-name deeplabv3_resnet101_deeplabv3_mobilenet_ssseg_mobilenetv2_cirkdv2 \
    --teacher-pretrained /data/winycg/cirkd/teachers/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base /home/sergey/DEV/AI/AEI/pretrained_backbones//mobilenet_v3_small-47085aa1.pth