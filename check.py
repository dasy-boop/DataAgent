import sys

print("解释器路径：", sys.executable)
print("是否使用虚拟环境：", sys.prefix != sys.base_prefix)
