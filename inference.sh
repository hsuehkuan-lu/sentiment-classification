dvc run -n inference --no-exec --force -d inference.py \
  -d data/test.csv -d "$VOCAB_PATH" -d "$CONFIG_PATH" \
  -d "$MODEL_PATH" -o "$SUBMISSION_PATH" \
  python inference.py