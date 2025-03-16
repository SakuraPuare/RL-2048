#!/usr/bin/env python3
"""
2048游戏回放工具

使用方法:
    python replay.py [记录文件] [--text] [--speed 速度]

参数:
    记录文件: 要回放的游戏记录文件路径
    --text: 在文本模式下回放（默认为图形界面）
    --speed: 回放速度倍数（默认为1.0）
"""

import sys
import argparse
from pathlib import Path
from src.core.game import Game2048
from src.visualization.display import ReplayDisplay

def main():
    parser = argparse.ArgumentParser(description='2048游戏回放工具')
    parser.add_argument('record_file', nargs='?', help='要回放的游戏记录文件')
    parser.add_argument('--text', action='store_true', help='在文本模式下回放（默认为图形界面）')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍数（默认为1.0）')
    args = parser.parse_args()
    
    # 创建record文件夹（如果不存在）
    record_dir = Path("record")
    record_dir.mkdir(exist_ok=True)
    
    # 如果没有提供记录文件，显示可用的记录文件
    if not args.record_file:
        # 从record文件夹中获取记录文件
        records = list(record_dir.glob('game_record_*.json'))
        
        if not records:
            print("未找到游戏记录文件。")
            print("请先玩游戏并保存记录，或指定记录文件路径。")
            return
        
        print("可用的游戏记录文件:")
        for i, record in enumerate(records, 1):
            print(f"{i}. {record.name}")
        
        try:
            choice = int(input("\n请选择要回放的记录（输入编号）: "))
            if 1 <= choice <= len(records):
                record_file = records[choice - 1]
            else:
                print("无效的选择。")
                return
        except ValueError:
            print("无效的输入。")
            return
    else:
        # 转换为Path对象
        record_file = Path(args.record_file)
        
        # 如果提供的文件名不包含路径，则添加record文件夹路径
        if not record_file.parent.name:
            record_file = record_dir / record_file
        
        # 如果文件名不以.json结尾，添加扩展名
        if record_file.suffix != '.json':
            record_file = record_file.with_suffix('.json')
    
    # 检查文件是否存在
    if not record_file.exists():
        print(f"错误：找不到记录文件 {record_file}")
        return
    
    try:
        # 加载记录
        replay_data = Game2048.replay_from_file(str(record_file))
        
        # 显示游戏信息
        print(f"\n游戏ID: {replay_data['game_id']}")
        print(f"开始时间: {replay_data['start_time']}")
        print(f"持续时间: {replay_data['duration']:.2f} 秒")
        print(f"最终分数: {replay_data['final_score']}")
        print(f"移动次数: {replay_data['moves_count']}")
        
        # 计算是否达到2048
        is_won = any(2048 in row for state in replay_data['history'] for row in state['grid'])
        print(f"是否达到2048: {'是' if is_won else '否'}")
        
        # 创建回放显示器
        display_type = "text" if args.text else "graphical"
        replay_display = ReplayDisplay(replay_data, display_type, args.speed)
        
        # 开始回放
        print(f"\n正在回放游戏记录：{record_file}")
        print(f"回放速度: {args.speed}x")
        
        if display_type == "graphical":
            print("控制：")
            print("  空格键 - 暂停/继续")
            print("  ESC    - 退出")
        
        input("\n按Enter开始回放...")
        replay_display.start_replay()
        
    except Exception as e:
        print(f"回放时出错：{e}")

if __name__ == "__main__":
    main() 