# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import hashlib
from pathlib import Path
from detector.config import Config


def generate_cache_path(data, filename: str, **kwargs):

    string = str(data) + "".join(map(str, kwargs.values()))
    hash_value = hashlib.md5(string.encode()).hexdigest()
    cache_dir = os.path.join(Config.CACHE_DIR, hash_value)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    return os.path.join(cache_dir, filename)


def count_py_lines_pathlib(directory):

    total_lines = 0
    file_count = 0

    exclude_dirs = {"ckpt", "__pycache__"}  # 你想排除的目录名集合

    for file_path in Path(directory).rglob("*.py"):  # 递归查找 .py 文件
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        with file_path.open("r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
            total_lines += line_count
            file_count += 1
            print(f"{file_path}: {line_count} 行")

    print(f"\nPython 文件总数: {file_count}")
    print(f"Python 代码总行数: {total_lines}")


if __name__ == "__main__":
    count_py_lines_pathlib(Config.PROJECT_DIR)
