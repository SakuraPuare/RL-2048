#!/usr/bin/env python3
"""
随机策略智能体，用于评估随机策略在2048游戏中的表现上限
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter
import sys
import os

# 添加2048目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '2048'))
from game import Game2048

# 设置matplotlib使用英文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RandomAgent:
    """随机策略智能体，在每一步随机选择一个有效动作"""
    
    def __init__(self):
        # 记录统计数据
        self.scores = []
        self.max_tiles = []
    
    def select_action(self, valid_actions):
        """随机选择一个有效动作"""
        return random.choice(valid_actions)
    
    def play_game(self, render=False, enable_recording=True):
        """玩一局游戏"""
        game = Game2048(enable_recording=enable_recording)
        
        if render:
            print("初始状态:")
            print(game.grid)
        
        while not game.is_game_over():
            # 获取有效动作
            valid_actions = self.get_valid_actions(game)
            
            # 选择动作
            action = self.select_action(valid_actions)
            
            # 执行动作
            game.move(action)
            
            if render:
                print(f"\n动作: {['上', '右', '下', '左'][action]}")
                print(f"分数: {game.score}")
                print(game.grid)
        
        # 记录结果
        max_tile = self.get_max_tile(game.grid)
        self.scores.append(game.score)
        self.max_tiles.append(max_tile)
        
        if render:
            print(f"\n游戏结束! 最终分数: {game.score}")
            print(f"最大方块: {max_tile}")
        
        # 保存游戏记录
        if enable_recording:
            record_file = game.save_record()
            if record_file and render:
                print(f"游戏记录已保存到: {record_file}")
        
        return game.score, max_tile
    
    def get_valid_actions(self, game):
        """获取当前游戏状态下的有效动作列表"""
        valid_actions = []
        
        # 创建游戏副本来测试动作
        for action in range(4):
            test_game = Game2048(enable_recording=False)
            test_game.grid = game.grid.copy()
            test_game.score = game.score
            
            # 如果动作改变了网格，则为有效动作
            if test_game.move(action):
                valid_actions.append(action)
        
        # 如果没有有效动作，返回所有动作（游戏将结束）
        if not valid_actions:
            valid_actions = list(range(4))
        
        return valid_actions
    
    def get_max_tile(self, grid):
        """获取网格上的最大方块值"""
        return np.max(grid)
    
    def evaluate(self, num_games=1000):
        """评估随机策略的性能"""
        self.scores = []
        self.max_tiles = []
        
        progress_bar = tqdm(range(num_games), desc="评估随机策略")
        
        for _ in progress_bar:
            score, max_tile = self.play_game(render=False, enable_recording=False)
            
            # 更新进度条
            progress_bar.set_postfix({
                '分数': score,
                '最大方块': max_tile
            })
        
        # 分析结果
        avg_score = np.mean(self.scores)
        max_score = np.max(self.scores)
        tile_counter = Counter(self.max_tiles)
        
        print(f"\n随机策略评估结果 ({num_games} 局游戏):")
        print(f"平均分数: {avg_score:.2f}")
        print(f"最高分数: {max_score}")
        print("\n最大方块分布:")
        
        # 按方块值排序
        for tile in sorted(tile_counter.keys()):
            percentage = (tile_counter[tile] / num_games) * 100
            print(f"方块 {tile}: {tile_counter[tile]} 次 ({percentage:.2f}%)")
        
        # 绘制结果
        self.plot_results(num_games)
        
        return avg_score, max_score, tile_counter
    
    def plot_results(self, num_games):
        """绘制评估结果"""
        plt.figure(figsize=(15, 5))
        
        # 绘制分数分布
        plt.subplot(1, 2, 1)
        plt.hist(self.scores, bins=20)
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        # 绘制最大方块分布
        plt.subplot(1, 2, 2)
        tile_values = sorted(list(set(self.max_tiles)))
        tile_counts = [self.max_tiles.count(v) for v in tile_values]
        
        # 转换为百分比
        tile_percentages = [(count / num_games) * 100 for count in tile_counts]
        
        plt.bar([str(v) for v in tile_values], tile_percentages)
        plt.title('Max Tile Distribution')
        plt.xlabel('Tile Value')
        plt.ylabel('Percentage (%)')
        
        plt.tight_layout()
        plt.savefig("random_agent_results.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='2048随机策略智能体')
    parser.add_argument('--play', action='store_true', help='玩一局游戏并显示过程')
    parser.add_argument('--evaluate', action='store_true', help='评估随机策略的性能')
    parser.add_argument('--games', type=int, default=1000, help='评估时的游戏局数')
    
    args = parser.parse_args()
    
    agent = RandomAgent()
    
    if args.play:
        print("使用随机策略玩一局游戏...")
        agent.play_game(render=True)
    
    if args.evaluate:
        print(f"评估随机策略 ({args.games} 局游戏)...")
        agent.evaluate(num_games=args.games)
    
    # 如果没有提供参数，则默认评估
    if not (args.play or args.evaluate):
        print("没有提供参数，默认评估随机策略 (1000 局游戏)...")
        agent.evaluate(num_games=1000)

if __name__ == "__main__":
    main() 