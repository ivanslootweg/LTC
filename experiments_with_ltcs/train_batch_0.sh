# 10-1
python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.000951 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --size 24 --model_id 2024071603 --reset

python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.000296 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --size 32 --model_id 2024071604 --reset
