export $(grep -v '^#' envs/.env | xargs)
dvc repro