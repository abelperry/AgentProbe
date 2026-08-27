#!/bin/bash
log_info() {
    echo -e "\033[32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# 1. 进入目录并执行单测
cd /workspace/code-agent-challenges/promotion_system/strategy
go test -v .

mkdir -p /logs/verifier
# 2. 根据结果给出最终判定
if [ $? -ne 0 ]; then
    log_error "Judge Result: FAILED"
    echo 0 > /logs/verifier/reward.txt
else
    log_info "Judge Result: PASSED"
    echo 1 > /logs/verifier/reward.txt
fi