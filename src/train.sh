# To run on the default GPU (physical ID 0)
python3 ./model_trainer.py \
    --model_type=VIT \
    --epochs=60 \
    --batch_size=1024 \
    --lr=1e-4 \
    --gpu_id=0
