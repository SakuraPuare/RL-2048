import sys
import argparse
from pathlib import Path
from src.core.game import Game2048
from src.visualization.display import ReplayDisplay
from src.utils.ensure_record_dir import ensure_record_dir

def main(interface_type='gui'):
    # 确保data/records文件夹存在
    ensure_record_dir()
    
    # 如果是从命令行直接运行，解析参数
    if interface_type is None:
        parser = argparse.ArgumentParser(description='2048游戏')
        parser.add_argument('--text', action='store_true', help='在文本模式下运行（默认：图形界面模式）')
        parser.add_argument('--replay', type=str, help='回放游戏记录文件')
        parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍数（默认：1.0）')
        args = parser.parse_args()
        
        interface_type = 'text' if args.text else 'gui'
        replay_file = args.replay
        replay_speed = args.speed
    else:
        replay_file = None
        replay_speed = 1.0
    
    # 回放模式
    if replay_file:
        # 转换为Path对象
        replay_file = Path(replay_file)
        
        # 如果提供的文件名不包含路径，则添加record文件夹路径
        if not replay_file.parent.name:
            record_dir = Path("record")
            replay_file = record_dir / replay_file
            
        # 如果文件名不以.json结尾，添加扩展名
        if replay_file.suffix != '.json':
            replay_file = replay_file.with_suffix('.json')
            
        if not replay_file.exists():
            print(f"错误：找不到记录文件 {replay_file}")
            return
        
        try:
            # 加载记录
            replay_data = Game2048.replay_from_file(str(replay_file))
            
            # 创建回放显示器
            display_type = "text" if interface_type == 'text' else "graphical"
            replay_display = ReplayDisplay(replay_data, display_type, replay_speed)
            
            # 开始回放
            print(f"正在回放游戏记录：{replay_file}")
            replay_display.start_replay()
            
        except Exception as e:
            print(f"回放时出错：{e}")
    
    # 正常游戏模式
    else:
        if interface_type == 'text':
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