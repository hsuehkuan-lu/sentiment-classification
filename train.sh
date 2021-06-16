dvc run -n train --no-exec --force -d train.py \
  -d data/train.csv -p train -p model \
  -o outputs/checkpoint.pth \
  -o outputs/config.json \
  -M outputs/results.json \
  python train.py