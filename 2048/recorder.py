import json
import time
import numpy as np
from datetime import datetime
import os
from pathlib import Path

class GameRecorder:
    def __init__(self):
        self.history = []
        self.start_time = time.time()
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建record文件夹
        self.record_dir = Path("record")
        self.record_dir.mkdir(exist_ok=True)
        
        # 设置默认文件名（在record文件夹中）
        self.current_filename = self.record_dir / f"game_record_{self.game_id}.json"
        
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
        self.current_filename.parent.mkdir(exist_ok=True)
        
        # 写入文件
        with open(self.current_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_to_file(self, filename=None):
        """
        将游戏历史保存到文件
        
        参数:
            filename: 文件名，如果为None则使用默认名称
        """
        if filename is not None:
            # 转换为Path对象
            filepath = Path(filename)
            
            # 如果提供的文件名不包含路径，则添加record文件夹路径
            if not filepath.parent.name:
                filepath = self.record_dir / filepath
            
            # 如果文件名不以.json结尾，添加扩展名
            if filepath.suffix != '.json':
                filepath = filepath.with_suffix('.json')
                
            # 更新当前文件名
            if filepath != self.current_filename:
                self.current_filename = filepath
                self._update_record_file()
        
        return str(self.current_filename)
    
    @staticmethod
    def load_from_file(filename):
        """
        从文件加载游戏历史
        
        参数:
            filename: 文件名
        
        返回:
            加载的游戏历史数据
        """
        # 转换为Path对象
        filepath = Path(filename)
        
        # 如果提供的文件名不包含路径，则添加record文件夹路径
        if not filepath.parent.name:
            record_dir = Path("record")
            filepath = record_dir / filepath
            
        # 如果文件名不以.json结尾，添加扩展名
        if filepath.suffix != '.json':
            filepath = filepath.with_suffix('.json')
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将网格数据转换回numpy数组
        for state in data["history"]:
            state["grid"] = np.array(state["grid"])
        
        return data 