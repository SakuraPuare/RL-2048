import sys
import pygame
from game import Game2048
from display import GraphicalDisplay, TEXT_COLOR_LIGHT

def main():
    game = Game2048(enable_recording=True)
    display = GraphicalDisplay()
    
    running = True
    saved_message = None
    saved_message_time = 0
    
    while running:
        current_time = pygame.time.get_ticks()
        
        # 处理事件
        for event in display.get_events():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # 游戏控制
                if not game.is_game_over():
                    if event.key in [pygame.K_UP, pygame.K_w]:
                        game.move(0)  # 上
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        game.move(1)  # 右
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        game.move(2)  # 下
                    elif event.key in [pygame.K_LEFT, pygame.K_a]:
                        game.move(3)  # 左
                    elif event.key == pygame.K_r:
                        # 手动保存记录到指定文件
                        filename = input("请输入保存文件名 (默认为自动生成): ")
                        if not filename:
                            filename = None
                        saved_filename = game.save_record(filename)
                        saved_message = f"游戏记录已保存到 {saved_filename}"
                        saved_message_time = current_time
        
        # 显示游戏
        display.display_game_state(
            game.get_grid(),
            game.get_score(),
            game.is_game_over(),
            game.is_won()
        )
        
        # 显示保存消息（如果有）
        if saved_message and current_time - saved_message_time < 3000:  # 显示3秒
            message_text = display.small_font.render(saved_message, True, TEXT_COLOR_LIGHT)
            display.screen.blit(
                message_text, 
                (display.screen.get_width() // 2 - message_text.get_width() // 2, 20)
            )
            pygame.display.flip()
        
        # 如果游戏结束，显示游戏结束消息
        if game.is_game_over() and not saved_message:
            saved_message = "游戏结束！录像已自动保存"
            saved_message_time = current_time
        
        display.tick(60)
    
    # 游戏结束时，确保录像已保存（实际上每一步都已经保存了）
    final_filename = game.save_record()
    print(f"游戏录像已保存到: {final_filename}")
    
    display.quit()

if __name__ == "__main__":
    main() 