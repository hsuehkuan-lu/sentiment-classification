dvc run -n prepare --no-exec --force \
  -d data/train.csv -p basic \
  -o outputs/vocab.plk \
  python prepare.py