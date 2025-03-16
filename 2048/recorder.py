import json
import time
import numpy as np
from datetime import datetime
import os

class GameRecorder:
    def __init__(self):
        self.history = []
        self.start_time = time.time()
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"game_record_{self.game_id}.json"
        
        # 创建初始记录文件
        self._update_record_file()
    
    def record_state(self, grid, score, move_direction=None):
        """
        记录游戏的一个状态
        
        参数:
            grid: 当前的游戏网格
            score: 当前分数
            move_direction: 导致此状态的移动方向 (0=上, 1=右, 2=下, 3=左, None=初始状态)
        """
        # 确保所有值都是JSON可序列化的
        grid_list = grid.tolist()  # 转换为列表以便JSON序列化
        score = int(score)  # 确保分数是Python原生int类型
        
        state = {
            "timestamp": float(time.time() - self.start_time),
            "grid": grid_list,
            "score": score,
            "move_direction": move_direction
        }
        self.history.append(state)
        
        # 每记录一步就更新文件
        self._update_record_file()
    
    def _update_record_file(self):
        """更新记录文件"""
        # 添加一些元数据
        data = {
            "game_id": self.game_id,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration": float(time.time() - self.start_time),
            "final_score": int(self.history[-1]["score"]) if self.history else 0,
            "moves_count": sum(1 for state in self.history if state["move_direction"] is not None),
            "history": self.history
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.current_filename) if os.path.dirname(self.current_filename) else '.', exist_ok=True)
        
        # 写入文件
        with open(self.current_filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_to_file(self, filename=None):
        """
        将游戏历史保存到文件
        
        参数:
            filename: 文件名，如果为None则使用默认名称
        """
        if filename is not None and filename != self.current_filename:
            # 如果指定了新文件名，则复制当前文件
            self.current_filename = filename
            self._update_record_file()
        
        return self.current_filename
    
    @staticmethod
    def load_from_file(filename):
        """
        从文件加载游戏历史
        
        参数:
            filename: 文件名
        
        返回:
            加载的游戏历史数据
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # 将网格数据转换回numpy数组
        for state in data["history"]:
            state["grid"] = np.array(state["grid"])
        
        return data 