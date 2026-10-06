# quat

一个只依赖 Python 标准库的四元数旋转内核：四元数乘法、共轭与逆、归一化、绕轴旋转、
两次旋转的串接、轴角与欧拉角互转，以及走短弧的球面插值。它不读写文件、不联网，
只在内存里处理浮点四元数。

## 目录

- quat/core.py：四元数内核
- tests/test_core.py：行为测试

## 跑测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v
