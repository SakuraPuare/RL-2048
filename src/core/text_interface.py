import sys
from src.core.game import Game2048
from src.visualization.display import TextDisplay

def main():
    game = Game2048(enable_recording=True)
    display = TextDisplay()
    
    while not game.is_game_over():
        # 显示游戏状态
        display.display_game_state(
            game.get_grid(),
            game.get_score(),
            game.is_game_over(),
            game.is_won()
        )
        
        # 获取用户输入
        key = input().lower()
        
        if key == 'q':
            print("感谢游玩！")
            # 退出前保存录像（实际上每一步都已经保存了）
            final_filename = game.save_record()
            print(f"游戏录像已保存到: {final_filename}")
            sys.exit(0)
        elif key == 'r':
            # 手动保存记录到指定文件
            filename = input("请输入保存文件名 (默认为自动生成): ")
            if not filename:
                filename = None
            saved_filename = game.save_record(filename)
            print(f"游戏记录已保存到 {saved_filename}")
            input("按Enter继续...")
        elif key in ['w', 'up']:
            game.move(0)  # 上
        elif key in ['d', 'right']:
            game.move(1)  # 右
        elif key in ['s', 'down']:
            game.move(2)  # 下
        elif key in ['a', 'left']:
            game.move(3)  # 左
    
    # 游戏结束
    display.display_game_state(
        game.get_grid(),
        game.get_score(),
        game.is_game_over(),
        game.is_won()
    )
    
    if game.is_won():
        print("\n恭喜！你达到了2048！")
    else:
        print("\n下次好运！")
    
    # 游戏结束时，确保录像已保存（实际上每一步都已经保存了）
    final_filename = game.save_record()
    print(f"游戏录像已保存到: {final_filename}")
    
    input("按Enter退出...")

if __name__ == "__main__":
    main() 