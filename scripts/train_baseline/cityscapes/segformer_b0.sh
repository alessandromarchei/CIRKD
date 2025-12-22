CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
    train_baseline_segformer.py \
    --model segformer \
    --backbone MiT_B0 \
    --dataset citys \
    --batch-size 8 \
    --workers 16 \
    --lr 0.0002 \
    --optimizer-type adamw \
    --crop-size 1024 1024 \
    --max-iterations 160000 \
    --data /home/sergey/DEV/AI/datasets/cityscapes/ \
    --save-dir checkpoints/baseline/cityscapes/ \
    --log-dir checkpoints/cirkd_v2/cityscapes/logs/ \
    --pretrained /data/winycg/seg_model_zoo/imagenet_backbone/mit/mit_b0.pth