#!/bin/bash

SESSION_NAME="monitor_sessions"

# Check whether the session exists and close it
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
fi

# Create a new session in the background
tmux new-session -s $SESSION_NAME -d

# First split into 2 columns (left/right)
tmux split-window -h -t $SESSION_NAME:0.0

# Left column: split into 3 rows
tmux split-window -v -t $SESSION_NAME:0.0
tmux split-window -v -t $SESSION_NAME:0.0

# Right column: split into 3 rows  
tmux split-window -v -t $SESSION_NAME:0.3
tmux split-window -v -t $SESSION_NAME:0.3

# Attach to each session in read-only mode (unset TMUX to allow nesting)
tmux send-keys -t $SESSION_NAME:0.0 "unset TMUX && tmux attach-session -t eeha_mnist_tests -r" C-m
tmux send-keys -t $SESSION_NAME:0.1 "unset TMUX && tmux attach-session -t eeha_mnist_tests_1 -r" C-m
tmux send-keys -t $SESSION_NAME:0.2 "unset TMUX && tmux attach-session -t eeha_mnist_tests_2 -r" C-m
tmux send-keys -t $SESSION_NAME:0.3 "unset TMUX && tmux attach-session -t eeha_mnist_tests_3 -r" C-m
tmux send-keys -t $SESSION_NAME:0.4 "unset TMUX && tmux attach-session -t eeha_mnist_tests_4 -r" C-m
tmux send-keys -t $SESSION_NAME:0.5 "unset TMUX && tmux attach-session -t eeha_mnist_tests_5 -r" C-m

# Attach to the monitoring session
tmux attach-session -t $SESSION_NAME
