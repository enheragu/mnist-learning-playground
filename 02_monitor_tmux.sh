#!/bin/bash

SESSION_NAME="monitor_sessions"

# Verificar si la sesión existe y cerrarla
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
fi

# Crear una nueva sesión en segundo plano
tmux new-session -s $SESSION_NAME -d

# Dividir la ventana horizontalmente
tmux split-window -h -t $SESSION_NAME -p 35

# Dividir el panel izquierdo verticalmente
tmux split-window -v -t $SESSION_NAME:0.0 -p 63

# Dividir el panel derecho verticalmente
tmux split-window -v -t $SESSION_NAME:0.1

# Ejecutar capture-pane de manera continua para cada panel
tmux send-keys -t $SESSION_NAME:0.0 "while true; do tmux capture-pane -t eeha_mnist_tests -pS -1000 | tail -n 20 | grep .; done" C-m
tmux send-keys -t $SESSION_NAME:0.1 "while true; do tmux capture-pane -t eeha_mnist_tests_2 -pS -1000 | tail -n 20 | grep .; done" C-m
tmux send-keys -t $SESSION_NAME:0.2 "while true; do tmux capture-pane -t eeha_mnist_tests_3 -pS -1000 | tail -n 20 | grep .; done" C-m

# Ejecutar watch -n 1 nvidia-smi en el cuarto panel
tmux send-keys -t $SESSION_NAME:0.3 "watch -n 1 nvidia-smi" C-m

# Organizar los paneles en una disposición de cuadrícula
# tmux select-layout -t $SESSION_NAME tiled

# Adjuntar a la sesión de monitoreo
tmux attach-session -t $SESSION_NAME
