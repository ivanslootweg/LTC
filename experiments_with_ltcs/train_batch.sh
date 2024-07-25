# 10-1
python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.000951 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --size 24 --model_id 2024071603 --reset

python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.000296 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --size 32 --model_id 2024071604 --reset

# 14-0
python3 forecast.py --epochs 200 --gpus 6 --initial_lr 0.00157 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --size 32 --model_id 2024071607 --reset

python3 forecast.py --epochs 200 --gpus 6 --initial_lr 0.000137 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --size 32 --model_id 2024071608 --reset
    
#14-1
python3 forecast.py --epochs 200 --gpus 7 --initial_lr 0.005 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 1 --future_loss --size 36 --model_id 2024071808 --reset

python3 forecast.py --epochs 200 --gpus 7 --initial_lr 0.022 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 1 --size 36 --model_id 2024071807 --reset

#13-0
python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.00181 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --iterative_forecast --size 48 --model_id 2024071907 --reset

python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.01 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --future_loss --iterative_forecast --size 36 --model_id 2024071903 --reset

#10-2
python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.01 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --iterative_forecast --size 48 --model_id 2024071810 --reset

python3 forecast.py --epochs 200 --gpus 2 --initial_lr 0.02 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 5 --iterative_forecast --size 36 --model_id 2024071811 --reset