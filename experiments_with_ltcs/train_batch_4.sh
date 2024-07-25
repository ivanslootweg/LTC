#10-2
# python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.01 --dataset neuronlaser \
#     --seq_len 32  --mixed_memory --future 5 --iterative_forecast --size 48 --model_id 2024071810 --reset

python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.02 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --iterative_forecast --size 36 --model_id 2024071811 --reset