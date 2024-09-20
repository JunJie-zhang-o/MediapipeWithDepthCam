#!/bin/bash

# 定义全局数组用于存储进程 ID
PIDS=()

# 定义启动程序并记录 PID 的函数
start_program() {
    local script=$1
    python3 "$script" &
    local pid=$!
    echo "Started $script with PID $pid"
    PIDS+=("$pid")
}

exec_program() {
    local script=$1
    $script &
    local pid=$!
    echo "Started $script with PID $pid"
    PIDS+=("$pid")
}

# 定义清理函数来终止所有记录的进程
cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill -SIGTERM "$pid" 2>/dev/null
        echo "killed program with PID $pid"
    done
}

# 捕获退出信号并调用 cleanup 函数
trap cleanup EXIT

# 气球 move until
# 请在运行前先启动机器人程序
start_program "main.py"
start_program "servoj.py"
exec_program "./rtplot.py"


# 等待所有后台进程完成
wait
