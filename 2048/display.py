import os
import sys
import time
import pygame
import numpy as np

# 颜色定义
GRID_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (237, 190, 30),
    8192: (237, 187, 14),
    16384: (237, 183, 0),
    32768: (237, 180, 0),
    65536: (237, 175, 0),
    131072: (237, 170, 0)
}

# 文本颜色
TEXT_COLOR_DARK = (119, 110, 101)  # 用于方块2和4
TEXT_COLOR_LIGHT = (249, 246, 242)  # 用于其他方块

# 游戏尺寸
GRID_SIZE = 4
CELL_SIZE = 100
GRID_PADDING = 10
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * GRID_PADDING
WINDOW_HEIGHT = WINDOW_WIDTH + 50  # 为分数留出额外空间

# 获取支持中文的字体
def get_font(size, bold=False):
    """获取支持中文的字体"""
    # 尝试使用常见的支持中文的字体
    chinese_fonts = [
        'SimHei',           # 中文黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 中文宋体
        'NSimSun',          # 新宋体
        'FangSong',         # 仿宋
        'KaiTi',            # 楷体
        'STHeiti',          # 华文黑体
        'STKaiti',          # 华文楷体
        'STSong',           # 华文宋体
        'STFangsong',       # 华文仿宋
        'Arial Unicode MS', # Arial Unicode
        'Noto Sans CJK SC', # Google Noto Sans CJK SC
        'Noto Sans SC',     # Google Noto Sans SC
        'WenQuanYi Micro Hei', # 文泉驿微米黑
        'Droid Sans Fallback', # Android Droid Sans Fallback
        'Arial'             # 最后尝试 Arial
    ]
    
    # 尝试每一个字体，直到找到系统上存在的字体
    for font_name in chinese_fonts:
        try:
            return pygame.font.SysFont(font_name, size, bold=bold)
        except:
            continue
    
    # 如果所有字体都失败，使用默认字体
    return pygame.font.SysFont(None, size, bold=bold)

# 辅助函数，用于计算游戏状态
def check_game_won(grid):
    """检查是否达到2048"""
    return np.any(grid >= 2048)

def check_game_over(grid):
    """检查游戏是否结束"""
    # 如果有空单元格，游戏未结束
    if np.any(grid == 0):
        return False
    
    # 检查相邻的相同方块
    for i in range(4):
        for j in range(3):
            if grid[i][j] == grid[i][j + 1]:
                return False
            if grid[j][i] == grid[j + 1][i]:
                return False
    
    return True

class TextDisplay:
    """文本界面显示器"""
    
    @staticmethod
    def clear_screen():
        """清除终端屏幕"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_grid(grid):
        """以格式化方式打印游戏网格"""
        print("+------+------+------+------+")
        for row in grid:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("      |", end="")
                else:
                    print(f"{cell:^6}|", end="")
            print("\n+------+------+------+------+")
    
    @staticmethod
    def print_instructions():
        """打印游戏说明"""
        print("\n控制:")
        print("  W 或 ↑ - 向上移动")
        print("  A 或 ← - 向左移动")
        print("  S 或 ↓ - 向下移动")
        print("  D 或 → - 向右移动")
        print("  Q     - 退出")
        print("  R     - 保存记录")
        print("\n按任意键继续...")
    
    @staticmethod
    def display_game_state(grid, score, is_game_over=None, is_won=None):
        """显示游戏状态"""
        # 如果未提供游戏状态，则计算
        if is_game_over is None:
            is_game_over = check_game_over(grid)
        if is_won is None:
            is_won = check_game_won(grid)
            
        TextDisplay.clear_screen()
        
        # 打印游戏状态
        print("\n" + "="*30)
        print(f"分数: {score}")
        print("="*30 + "\n")
        
        TextDisplay.print_grid(grid)
        
        if is_won:
            print("\n恭喜！你已经达到了2048！")
            print("你可以继续游戏以获得更高分数。")
        
        if is_game_over:
            print("\n游戏结束！")
        
        TextDisplay.print_instructions()


class GraphicalDisplay:
    """图形界面显示器"""
    
    def __init__(self):
        """初始化pygame和显示设置"""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2048")
        self.font = get_font(40, bold=True)
        self.small_font = get_font(24)
        self.clock = pygame.time.Clock()
    
    def draw_grid(self, grid):
        """绘制游戏网格"""
        # 绘制背景
        self.screen.fill(GRID_COLOR)
        
        # 绘制单元格
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = grid[row][col]
                
                # 计算位置
                x = col * CELL_SIZE + (col + 1) * GRID_PADDING
                y = row * CELL_SIZE + (row + 1) * GRID_PADDING
                
                # 绘制单元格背景
                pygame.draw.rect(
                    self.screen,
                    TILE_COLORS.get(value, TILE_COLORS[0]),
                    (x, y, CELL_SIZE, CELL_SIZE),
                    border_radius=5
                )
                
                # 如果单元格不为空，则绘制值文本
                if value != 0:
                    # 根据方块值选择文本颜色
                    text_color = TEXT_COLOR_DARK if value in [2, 4] else TEXT_COLOR_LIGHT
                    
                    # 根据数字位数调整字体大小
                    font_size = 40
                    if value > 1000:
                        font_size = 30
                    if value > 10000:
                        font_size = 24
                    
                    font = get_font(font_size, bold=True)
                    text = font.render(str(value), True, text_color)
                    
                    # 在单元格中居中文本
                    text_rect = text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    self.screen.blit(text, text_rect)
    
    def draw_score(self, score, is_game_over=None, is_won=None):
        """绘制分数和游戏状态"""
        # 绘制分数
        score_text = self.small_font.render(f"分数: {score}", True, TEXT_COLOR_LIGHT)
        self.screen.blit(score_text, (10, WINDOW_HEIGHT - 40))
        
        # 绘制游戏状态
        if is_won:
            status_text = self.small_font.render("你已经达到2048！继续前进！", True, TEXT_COLOR_LIGHT)
            self.screen.blit(status_text, (WINDOW_WIDTH // 2 - status_text.get_width() // 2, WINDOW_HEIGHT - 40))
        elif is_game_over:
            status_text = self.small_font.render("游戏结束！", True, TEXT_COLOR_LIGHT)
            self.screen.blit(status_text, (WINDOW_WIDTH // 2 - status_text.get_width() // 2, WINDOW_HEIGHT - 40))
    
    def display_game_state(self, grid, score, is_game_over=None, is_won=None):
        """显示完整的游戏状态"""
        # 如果未提供游戏状态，则计算
        if is_game_over is None:
            is_game_over = check_game_over(grid)
        if is_won is None:
            is_won = check_game_won(grid)
            
        self.draw_grid(grid)
        self.draw_score(score, is_game_over, is_won)
        pygame.display.flip()
    
    def get_events(self):
        """获取pygame事件"""
        return pygame.event.get()
    
    def tick(self, fps=60):
        """控制帧率"""
        self.clock.tick(fps)
    
    def quit(self):
        """退出pygame"""
        pygame.quit()


class ReplayDisplay:
    """回放显示器，用于显示记录的游戏"""
    
    def __init__(self, replay_data, display_type="graphical", speed=1.0):
        """
        初始化回放显示器
        
        参数:
            replay_data: 从文件加载的回放数据
            display_type: 显示类型 ("text" 或 "graphical")
            speed: 回放速度倍数（默认为1.0，影响每步之间的等待时间）
        """
        self.replay_data = replay_data
        self.display_type = display_type
        self.speed = speed
        
        if display_type == "graphical":
            self.display = GraphicalDisplay()
        else:
            self.display = TextDisplay()
    
    def start_replay(self):
        """开始回放游戏"""
        history = self.replay_data["history"]
        
        if self.display_type == "graphical":
            self._graphical_replay(history)
        else:
            self._text_replay(history)
    
    def _graphical_replay(self, history):
        """图形界面回放"""
        running = True
        current_index = 0
        
        # 每步之间的固定等待时间（秒）
        step_delay = 0.5 / self.speed  # 默认0.5秒，根据速度调整
        
        while running and current_index < len(history):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # 空格键暂停/继续
                        paused = True
                        while paused and running:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                if pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_SPACE:
                                        paused = False
                                    elif pause_event.key == pygame.K_ESCAPE:
                                        running = False
                                        paused = False
                                    elif pause_event.key == pygame.K_RIGHT:
                                        # 右箭头键：下一步
                                        paused = False
                                        step_delay = 0  # 立即显示下一步
                                    elif pause_event.key == pygame.K_LEFT and current_index > 0:
                                        # 左箭头键：上一步
                                        current_index = max(0, current_index - 2)
                                        paused = False
                                        step_delay = 0  # 立即显示
                            
                            # 显示暂停消息
                            pause_text = get_font(24).render("已暂停 - 按空格键继续", True, TEXT_COLOR_LIGHT)
                            self.display.screen.blit(
                                pause_text, 
                                (WINDOW_WIDTH // 2 - pause_text.get_width() // 2, 20)
                            )
                            pygame.display.flip()
                            self.display.tick(30)
                    elif event.key == pygame.K_RIGHT:
                        # 右箭头键：加速
                        step_delay = 0  # 立即显示下一步
                    elif event.key == pygame.K_LEFT and current_index > 0:
                        # 左箭头键：上一步
                        current_index = max(0, current_index - 2)
                        step_delay = 0  # 立即显示
            
            # 获取当前状态
            state = history[current_index]
            grid = state["grid"]
            score = state["score"]
            
            # 计算游戏状态
            is_won = check_game_won(grid)
            is_game_over = current_index == len(history) - 1 and check_game_over(grid)
            
            # 显示当前状态
            self.display.display_game_state(
                grid,
                score,
                is_game_over,
                is_won
            )
            
            # 显示步骤信息
            step_info = f"步骤: {current_index + 1}/{len(history)}"
            if state["move_direction"] is not None:
                direction_names = ["上", "右", "下", "左"]
                step_info += f" - 移动: {direction_names[state['move_direction']]}"
            
            step_text = get_font(18).render(step_info, True, TEXT_COLOR_LIGHT)
            self.display.screen.blit(
                step_text, 
                (10, 10)
            )
            pygame.display.flip()
            
            # 等待固定时间
            if step_delay > 0:
                time.sleep(step_delay)
            else:
                # 如果是由于按键导致的立即显示，重置延迟时间
                step_delay = 0.5 / self.speed
            
            current_index += 1
            self.display.tick(60)
        
        # 显示最终状态一段时间
        if current_index >= len(history) and running:
            final_state = history[-1]
            grid = final_state["grid"]
            score = final_state["score"]
            
            # 计算最终游戏状态
            is_won = check_game_won(grid)
            is_game_over = check_game_over(grid)
            
            self.display.display_game_state(
                grid,
                score,
                is_game_over,
                is_won
            )
            
            # 显示回放结束消息
            end_text = get_font(24).render("回放结束 - 按ESC退出", True, TEXT_COLOR_LIGHT)
            self.display.screen.blit(
                end_text, 
                (WINDOW_WIDTH // 2 - end_text.get_width() // 2, 20)
            )
            pygame.display.flip()
            
            # 等待用户按ESC退出
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        waiting = False
                self.display.tick(30)
        
        self.display.quit()
    
    def _text_replay(self, history):
        """文本界面回放"""
        current_index = 0
        
        while current_index < len(history):
            # 获取当前状态
            state = history[current_index]
            grid = state["grid"]
            score = state["score"]
            
            # 计算游戏状态
            is_won = check_game_won(grid)
            is_game_over = current_index == len(history) - 1 and check_game_over(grid)
            
            # 显示当前状态
            self.display.display_game_state(
                grid,
                score,
                is_game_over,
                is_won
            )
            
            # 显示步骤信息
            print(f"\n回放进度: {current_index + 1}/{len(history)}")
            if state["move_direction"] is not None:
                direction_names = ["上", "右", "下", "左"]
                print(f"移动方向: {direction_names[state['move_direction']]}")
            
            print("按Enter继续，按Q退出，按B返回上一步")
            
            # 等待用户输入
            user_input = input().lower()
            if user_input == 'q':
                break
            elif user_input == 'b' and current_index > 0:
                current_index -= 1
                continue
            
            current_index += 1
        
        # 显示最终状态
        if current_index >= len(history):
            final_state = history[-1]
            grid = final_state["grid"]
            score = final_state["score"]
            
            # 计算最终游戏状态
            is_won = check_game_won(grid)
            is_game_over = check_game_over(grid)
            
            self.display.display_game_state(
                grid,
                score,
                is_game_over,
                is_won
            )
            print("\n回放结束！")
            input("按Enter退出") 