EXPERIMENT="sum_of_features"
CSV="data/train.csv"
GPUS="0,"

python NeuralNetwork/train.py \
    --experiment ${EXPERIMENT} \
    --data_csv ${CSV} \
    --num_workers=16 \
    --gpus=${GPUS} \
    --batch_size=8192 \
    --check_val_every_n_epoch=5 \
    --val_size=0.25 \
    --input_dimentions=1