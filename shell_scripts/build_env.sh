dvc run -n build --force \
  --always-changed \
  export $(grep -v '^#' envs/.env | xargs)