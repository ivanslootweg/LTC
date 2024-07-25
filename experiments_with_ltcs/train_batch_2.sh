    
#14-1
# python3 forecast.py --epochs 200 --gpus 7 --initial_lr 0.005 --dataset neuronlaser \
#     --seq_len 32  --mixed_memory --future 1 --future_loss --size 36 --model_id 2024071808 --reset

python3 forecast.py --epochs 200 --gpus 7 --initial_lr 0.022 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 1 --size 36 --model_id 2024071807 --reset


#14-1
python3 forecast.py --epochs 400 --gpus 7 --initial_lr 0.001 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 1 --future_loss --size 32 --model_id 2024072302
#14-3
python3 forecast.py --epochs 400 --gpus 7 --initial_lr 0.001 --dataset neuronlaser \
    --seq_len 32  --mixed_memory --future 1 --size 32 --model_id_shift -3