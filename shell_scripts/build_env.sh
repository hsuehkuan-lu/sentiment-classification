dvc run -n build --force \
  --always-changed \
  export $(grep -v '^#' .env | xargs)