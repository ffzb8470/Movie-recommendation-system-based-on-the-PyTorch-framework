import sys
msg = sys.stdin.read().strip()
lines = msg.split('\n')
title = lines[0].strip()
# Clear specific commit titles
targets = [
    '优化：清理无用文件 & Windows 兼容性修复',
    '更新 README.md：添加项目描述、技术栈、使用说明',
    '添加 .gitignore，排除 __pycache__、模型文件、日志、训练图片',
]
if title in targets:
    print('.')
else:
    print(msg)