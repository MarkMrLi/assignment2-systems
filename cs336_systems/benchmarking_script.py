"""
Benchmark 入口脚本 (向后兼容)

这个脚本是对新模块化实现的简单封装。
实际逻辑在 benchmark_cli.py 中。

使用方式:
    python -m cs336_systems.benchmarking_script [options]

或者直接使用新接口:
    python -m cs336_systems.benchmark_cli [options]
"""

from .benchmark_cli import main

if __name__ == "__main__":
    main()
