import sys
import argparse
import os
from game import Game2048
from display import ReplayDisplay

def main():
    parser = argparse.ArgumentParser(description='2048游戏')
    parser.add_argument('--text', action='store_true', help='在文本模式下运行（默认：图形界面模式）')
    parser.add_argument('--replay', type=str, help='回放游戏记录文件')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍数（默认：1.0）')
    args = parser.parse_args()
    
    # 回放模式
    if args.replay:
        if not os.path.exists(args.replay):
            print(f"错误：找不到记录文件 {args.replay}")
            return
        
        try:
            # 加载记录
            replay_data = Game2048.replay_from_file(args.replay)
            
            # 创建回放显示器
            display_type = "text" if args.text else "graphical"
            replay_display = ReplayDisplay(replay_data, display_type, args.speed)
            
            # 开始回放
            print(f"正在回放游戏记录：{args.replay}")
            replay_display.start_replay()
            
        except Exception as e:
            print(f"回放时出错：{e}")
    
    # 正常游戏模式
    else:
        if args.text:
            # 运行文本界面
            from text_interface import main as text_main
            text_main()
        else:
            try:
                # 尝试运行图形界面
                from gui_interface import main as gui_main
                gui_main()
            except ImportError:
                print("Pygame未安装。改为在文本模式下运行。")
                print("要安装Pygame，请运行：pip install pygame")
                print("或使用--text标志有意使用文本模式。")
                
                # 回退到文本界面
                from text_interface import main as text_main
                text_main()

if __name__ == "__main__":
    main() 