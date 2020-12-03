EXPERIMENT="baseline"
CSV="data/train.csv"

python NeuralNetwork/train.py \
    --experiment ${EXPERIMENT} \
    --data_csv ${CSV} \
    --num_workers=16 \
    --batch_size=256 \
    --check_val_every_n_epoch=5 \
    --val_size=0.25