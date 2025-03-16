import random
import numpy as np
from src.utils.recorder import GameRecorder

class Game2048:
    def __init__(self, enable_recording=True):
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        self.won = False
        self.enable_recording = enable_recording
        
        # 初始化记录器
        if self.enable_recording:
            self.recorder = GameRecorder()
        
        # 添加两个初始方块
        self.add_new_tile()
        self.add_new_tile()
        
        # 记录初始状态
        if self.enable_recording:
            self.recorder.record_state(
                self.grid.copy(), 
                int(self.score), 
                move_direction=None
            )
    
    def add_new_tile(self):
        """在随机空单元格中添加一个新方块（2或4）"""
        if np.count_nonzero(self.grid == 0) == 0:
            return False
        
        # 找到空单元格
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if not empty_cells:
            return False
        
        # 选择一个随机空单元格
        row, col = random.choice(empty_cells)
        
        # 放置一个2（90%几率）或4（10%几率）
        self.grid[row, col] = 2 if random.random() < 0.9 else 4
        return True
    
    def move(self, direction):
        """
        向指定方向移动方块
        direction: 0=上, 1=右, 2=下, 3=左
        """
        if self.game_over:
            return False
        
        # 保存当前状态以检查移动是否改变了什么
        prev_grid = self.grid.copy()
        prev_score = self.score
        
        # 根据方向处理网格
        if direction == 0:  # 上
            self._move_up()
        elif direction == 1:  # 右
            self._move_right()
        elif direction == 2:  # 下
            self._move_down()
        elif direction == 3:  # 左
            self._move_left()
        
        # 检查移动是否改变了网格
        grid_changed = not np.array_equal(self.grid, prev_grid)
        
        if grid_changed:
            self.add_new_tile()
            
            # 检查游戏是否结束
            if not self.can_move():
                self.game_over = True
            
            # 只有当操作改变了网格时才记录状态
            if self.enable_recording:
                self.recorder.record_state(
                    self.grid.copy(), 
                    int(self.score), 
                    move_direction=int(direction) if direction is not None else None
                )
            
            return True
        
        return False
    
    def _move_left(self):
        """向左移动方块并在可能的情况下合并"""
        for i in range(4):
            # 获取行
            row = self.grid[i]
            # 合并和压缩行
            new_row = self._merge_row(row)
            # 更新网格
            self.grid[i] = new_row
    
    def _move_right(self):
        """向右移动方块并在可能的情况下合并"""
        for i in range(4):
            # 获取行，反转它，合并，然后再反转回来
            row = self.grid[i][::-1]
            # 合并和压缩行
            new_row = self._merge_row(row)
            # 更新网格（反转回来）
            self.grid[i] = new_row[::-1]
    
    def _move_up(self):
        """向上移动方块并在可能的情况下合并"""
        for j in range(4):
            # 获取列
            col = self.grid[:, j]
            # 合并和压缩列
            new_col = self._merge_row(col)
            # 更新网格
            self.grid[:, j] = new_col
    
    def _move_down(self):
        """向下移动方块并在可能的情况下合并"""
        for j in range(4):
            # 获取列，反转它，合并，然后再反转回来
            col = self.grid[:, j][::-1]
            # 合并和压缩列
            new_col = self._merge_row(col)
            # 更新网格（反转回来）
            self.grid[:, j] = new_col[::-1]
    
    def _merge_row(self, row):
        """
        合并和压缩单行或单列
        这处理移动和合并方块的核心逻辑
        """
        # 移除零并获取非零值
        row = row[row != 0]
        
        # 初始化结果
        result = []
        i = 0
        
        # 处理行
        while i < len(row):
            # 如果这不是最后一个元素，并且当前元素等于下一个元素
            if i + 1 < len(row) and row[i] == row[i + 1]:
                # 合并方块
                merged_value = row[i] * 2
                result.append(merged_value)
                # 更新分数
                self.score += int(merged_value)
                # 检查胜利条件
                if merged_value == 2048 and not self.won:
                    self.won = True
                # 跳过下一个元素，因为它已经被合并
                i += 2
            else:
                # 只添加当前元素
                result.append(row[i])
                i += 1
        
        # 用零填充以保持网格大小
        result = result + [0] * (4 - len(result))
        return np.array(result)
    
    def can_move(self):
        """检查是否可能有任何移动"""
        # 如果有空单元格，则可以移动
        if np.any(self.grid == 0):
            return True
        
        # 检查相邻的相同方块
        for i in range(4):
            for j in range(3):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True
                if self.grid[j][i] == self.grid[j + 1][i]:
                    return True
        
        return False
    
    def get_grid(self):
        return self.grid
    
    def get_score(self):
        return int(self.score)
    
    def is_game_over(self):
        return bool(self.game_over)
    
    def is_won(self):
        return bool(self.won)
    
    def save_record(self, filename=None):
        """保存游戏记录到文件"""
        if not self.enable_recording:
            return None
        
        return self.recorder.save_to_file(filename)
    
    @staticmethod
    def replay_from_file(filename):
        """从文件加载游戏记录"""
        from recorder import GameRecorder
        return GameRecorder.load_from_file(filename) 