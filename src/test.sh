# To run on the default GPU (physical ID 0)
python3 ./model_runner.py \
    --model_path=../model_weight/CNN_20250817_193800.pth \
    --model_type=CNN \
    --batch_size=1024 \
    --gpu_id=0
