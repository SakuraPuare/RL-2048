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
import os
import argparse
from game import Game2048
from display import ReplayDisplay

def main():
    parser = argparse.ArgumentParser(description='2048游戏回放工具')
    parser.add_argument('record_file', nargs='?', help='要回放的游戏记录文件')
    parser.add_argument('--text', action='store_true', help='在文本模式下回放（默认为图形界面）')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍数（默认为1.0）')
    args = parser.parse_args()
    
    # 如果没有提供记录文件，显示可用的记录文件
    if not args.record_file:
        records = [f for f in os.listdir('.') if f.startswith('game_record_') and f.endswith('.json')]
        
        if not records:
            print("未找到游戏记录文件。")
            print("请先玩游戏并保存记录，或指定记录文件路径。")
            return
        
        print("可用的游戏记录文件:")
        for i, record in enumerate(records, 1):
            print(f"{i}. {record}")
        
        try:
            choice = int(input("\n请选择要回放的记录（输入编号）: "))
            if 1 <= choice <= len(records):
                args.record_file = records[choice - 1]
            else:
                print("无效的选择。")
                return
        except ValueError:
            print("无效的输入。")
            return
    
    # 检查文件是否存在
    if not os.path.exists(args.record_file):
        print(f"错误：找不到记录文件 {args.record_file}")
        return
    
    try:
        # 加载记录
        replay_data = Game2048.replay_from_file(args.record_file)
        
        # 显示游戏信息
        print(f"\n游戏ID: {replay_data['game_id']}")
        print(f"开始时间: {replay_data['start_time']}")
        print(f"持续时间: {replay_data['duration']:.2f} 秒")
        print(f"最终分数: {replay_data['final_score']}")
        print(f"移动次数: {replay_data['moves_count']}")
        print(f"是否达到2048: {'是' if replay_data['is_won'] else '否'}")
        
        # 创建回放显示器
        display_type = "text" if args.text else "graphical"
        replay_display = ReplayDisplay(replay_data, display_type, args.speed)
        
        # 开始回放
        print(f"\n正在回放游戏记录：{args.record_file}")
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