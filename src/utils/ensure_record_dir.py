#!/usr/bin/env python3
"""
确保data/records文件夹存在的辅助脚本
"""

import sys
from pathlib import Path

def ensure_record_dir():
    """确保data/records文件夹存在"""
    record_dir = Path("data/records")
    
    # 如果data/records文件夹不存在，则创建它
    if not record_dir.exists():
        try:
            record_dir.mkdir(parents=True, exist_ok=True)
            print(f"已创建 {record_dir} 文件夹")
        except Exception as e:
            print(f"创建 {record_dir} 文件夹时出错: {e}")
            return False
    
    # 检查data/records文件夹是否可写
    try:
        # 尝试创建一个临时文件来测试写入权限
        test_file = record_dir / ".write_test"
        test_file.touch()
        test_file.unlink()  # 删除测试文件
    except (PermissionError, OSError):
        print(f"警告: {record_dir} 文件夹不可写")
        return False
    
    return True

if __name__ == "__main__":
    if ensure_record_dir():
        print("data/records文件夹已准备就绪")
    else:
        print("无法确保data/records文件夹存在或可写")
        sys.exit(1) 