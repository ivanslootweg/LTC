#13-0
python3 forecast.py --epochs 200 --gpus 0 --initial_lr 0.00181 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --iterative_forecast --size 48 --model_id 2024071907 --reset

python3 forecast.py --epochs 200 --gpus 0 --initial_lr 0.01 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --iterative_forecast --size 36 --model_id 2024071903 --reset
