dvc run -n train --no-exec --force -d train.py \
  -d $VOCAB_PATH -d data/train.csv \
  -p train -p model \
  -o $MODEL_PATH \
  -o $CONFIG_PATH \
  -M $RESULTS_PATH \
  python train.py