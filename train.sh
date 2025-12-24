#!/usr/bin/env bash
set -e

PID=2138420              # <-- metti qui il PID del training in corso
SLEEP=60               # secondi tra i check

echo "Waiting for process $PID to finish..."

while kill -0 "$PID" 2>/dev/null; do
    sleep "$SLEEP"
done

echo "Process $PID finished. Starting training."



#training to do : deeplab v3 + efficientnet b0, b1
bash scripts/train_baseline/cityscapes/deeplabv3_efficientnetb2_aspp512_4.sh
bash scripts/train_baseline/cityscapes/deeplabv3_efficientnetb4_4.sh
bash scripts/train_baseline/cityscapes/deeplabv3_efficientnetb3_aspp512_4.sh
bash scripts/train_baseline/cityscapes/deeplabv3_efficientnetb6.sh
# bash scripts/train_baseline/cityscapes/deeplabv3_efficientnetb2_aspp512.sh