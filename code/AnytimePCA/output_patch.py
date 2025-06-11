"""
AnytimePCA输出路径补丁

这个脚本会修改AnytimePCA的输出路径，使其输出到指定目录
使用方法：
1. 将此文件放在AnytimePCA目录下
2. 在运行AnytimePCA前，设置环境变量SSPCA_OUT_DIR为期望的输出目录
"""

import os
import sys
import builtins
import importlib.util
import shutil
from datetime import datetime

# 获取sspca_exp.py的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sspca_path = os.path.join(script_dir, 'sspca_exp.py')

# 检查文件是否存在
if not os.path.exists(sspca_path):
    print(f"错误: 找不到文件 {sspca_path}")
    sys.exit(1)

# 保存原始命令行参数
original_args = sys.argv.copy()

# 修改输出目录
custom_out_dir = None
if 'SSPCA_OUT_DIR' in os.environ:
    custom_out_dir = os.environ['SSPCA_OUT_DIR']
    print(f"使用自定义输出目录: {custom_out_dir}")
    os.makedirs(custom_out_dir, exist_ok=True)
    
    # 创建日期子目录
    date_dir = os.path.join(custom_out_dir, datetime.now().strftime("%y_%m_%d"))
    os.makedirs(date_dir, exist_ok=True)
    print(f"创建日期目录: {date_dir}")
else:
    print("未设置环境变量SSPCA_OUT_DIR，使用默认输出目录")

# 直接修改sspca_exp.py文件
def patch_sspca_file():
    if not custom_out_dir:
        return False
        
    # 创建备份
    backup_path = sspca_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(sspca_path, backup_path)
        print(f"已创建备份: {backup_path}")
    
    # 读取原始文件
    with open(sspca_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修改过
    patch_marker = "# PATCHED_FOR_CUSTOM_OUTPUT_DIR"
    if patch_marker in content:
        print("文件已经被修改过，不再重复修改")
        return True
    
    # 查找log_dir定义的行
    log_dir_line = None
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'log_dir' in line and '=' in line and ('os.path.join' in line or 'PROJECT_DIR' in line):
            log_dir_line = i
            break
    
    if log_dir_line is not None:
        # 替换log_dir定义，使用原始字符串r前缀避免转义问题
        lines[log_dir_line] = f'log_dir = r"{custom_out_dir}" {patch_marker}'
        print(f"修改第 {log_dir_line+1} 行: {lines[log_dir_line]}")
        
        # 写回文件
        with open(sspca_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("文件修改成功")
        return True
    else:
        print("未找到log_dir定义，无法修改文件")
        return False

# 尝试修改文件
patched = patch_sspca_file()

# 直接运行原始脚本
if __name__ == "__main__":
    # 准备命令行参数
    sys.argv[0] = sspca_path  # 替换脚本名称
    
    # 设置环境变量，确保子进程也能访问
    if custom_out_dir:
        os.environ['SSPCA_OUT_DIR'] = r"{}".format(custom_out_dir)
    
    # 直接执行原始脚本
    print(f"执行: {sspca_path} {' '.join(sys.argv[1:])}")
    
    # 使用subprocess运行脚本，这样可以捕获输出
    import subprocess
    result = subprocess.run([sys.executable, sspca_path] + sys.argv[1:], 
                           capture_output=True, text=True)
    
    # 打印输出
    print("===== 标准输出 =====")
    print(result.stdout)
    
    if result.stderr:
        print("===== 错误输出 =====")
        print(result.stderr)
    
    # 返回原始脚本的退出码
    sys.exit(result.returncode) 