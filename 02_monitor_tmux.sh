#!/bin/bash

SESSION_NAME="monitor_sessions"

# Verificar si la sesión existe y cerrarla
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
fi

# Crear una nueva sesión en segundo plano
tmux new-session -s $SESSION_NAME -d

# Dividir primero en 2 columnas (izq/der)
tmux split-window -h -t $SESSION_NAME:0.0

# Columna izquierda: dividir en 3 filas
tmux split-window -v -t $SESSION_NAME:0.0
tmux split-window -v -t $SESSION_NAME:0.0

# Columna derecha: dividir en 3 filas  
tmux split-window -v -t $SESSION_NAME:0.3
tmux split-window -v -t $SESSION_NAME:0.3

# Ejecutar attach en modo solo lectura a cada sesión (unset TMUX para permitir anidamiento)
tmux send-keys -t $SESSION_NAME:0.0 "unset TMUX && tmux attach-session -t eeha_mnist_tests -r" C-m
tmux send-keys -t $SESSION_NAME:0.1 "unset TMUX && tmux attach-session -t eeha_mnist_tests_1 -r" C-m
tmux send-keys -t $SESSION_NAME:0.2 "unset TMUX && tmux attach-session -t eeha_mnist_tests_2 -r" C-m
tmux send-keys -t $SESSION_NAME:0.3 "unset TMUX && tmux attach-session -t eeha_mnist_tests_3 -r" C-m
tmux send-keys -t $SESSION_NAME:0.4 "unset TMUX && tmux attach-session -t eeha_mnist_tests_4 -r" C-m
tmux send-keys -t $SESSION_NAME:0.5 "unset TMUX && tmux attach-session -t eeha_mnist_tests_5 -r" C-m

# Adjuntar a la sesión de monitoreo
tmux attach-session -t $SESSION_NAME
