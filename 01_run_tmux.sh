#!/usr/bin/env bash

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTFILE_PATH=$SOURCE
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

VENV_PATH="$SCRIPT_PATH/../venv/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
  echo "ERROR: Could not find a valid virtual env in: $VENV_PATH"
  exit 1
fi

TMUX_CMD="source \"$VENV_PATH\" && cd \"$SCRIPT_PATH\" && python src/01_train_ablation_test.py"

tmux new-session -d -s "eeha_mnist_tests" "$TMUX_CMD"
tmux new-session -d -s "eeha_mnist_tests_2" "$TMUX_CMD"
tmux new-session -d -s "eeha_mnist_tests_3" "$TMUX_CMD"