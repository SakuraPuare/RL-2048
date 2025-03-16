#!/usr/bin/env python3
"""
随机策略智能体，用于评估随机策略在2048游戏中的表现上限
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse
from collections import Counter
import sys
import os
import multiprocessing
import time

# Import from the new structure
from src.core.game import Game2048

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
    
    def evaluate_parallel(self, num_games=100000, num_processes=None):
        """使用多进程并行评估随机策略的性能"""
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        print(f"使用 {num_processes} 个进程并行评估随机策略 ({num_games} 局游戏)...")
        start_time = time.time()
        
        # 使用process_map并行运行游戏
        results = process_map(play_game_worker, range(num_games), 
                             max_workers=num_processes, 
                             desc="并行评估随机策略",
                             chunksize=max(1, num_games // (num_processes * 10)))
        
        # 收集结果
        scores, max_tiles = zip(*results)
        
        # 分析结果
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        tile_counter = Counter(max_tiles)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n随机策略并行评估结果 ({num_games} 局游戏):")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每局耗时: {elapsed_time/num_games*1000:.2f} 毫秒")
        print(f"平均分数: {avg_score:.2f}")
        print(f"最高分数: {max_score}")
        print("\n最大方块分布:")
        
        # 按方块值排序
        for tile in sorted(tile_counter.keys()):
            percentage = (tile_counter[tile] / num_games) * 100
            print(f"方块 {tile}: {tile_counter[tile]} 次 ({percentage:.4f}%)")
        
        # 绘制结果
        self.scores = list(scores)
        self.max_tiles = list(max_tiles)
        self.plot_results(num_games)
        
        # 保存结果到文件
        self.save_results_to_file(scores, max_tiles, num_games)
        
        return avg_score, max_score, tile_counter
    
    def save_results_to_file(self, scores, max_tiles, num_games):
        """保存评估结果到文件"""
        # 保存分数和最大方块值
        np.savez(f"random_agent_results_{num_games}.npz", 
                 scores=np.array(scores), 
                 max_tiles=np.array(max_tiles))
        
        # 保存统计信息到文本文件
        with open(f"random_agent_stats_{num_games}.txt", "w") as f:
            f.write(f"随机策略评估结果 ({num_games} 局游戏):\n")
            f.write(f"平均分数: {np.mean(scores):.2f}\n")
            f.write(f"最高分数: {np.max(scores)}\n")
            f.write(f"分数标准差: {np.std(scores):.2f}\n")
            f.write(f"分数中位数: {np.median(scores):.2f}\n")
            f.write("\n最大方块分布:\n")
            
            tile_counter = Counter(max_tiles)
            for tile in sorted(tile_counter.keys()):
                percentage = (tile_counter[tile] / num_games) * 100
                f.write(f"方块 {tile}: {tile_counter[tile]} 次 ({percentage:.4f}%)\n")
    
    def plot_results(self, num_games):
        """绘制评估结果"""
        plt.figure(figsize=(15, 10))
        
        # 绘制分数分布
        plt.subplot(2, 2, 1)
        plt.hist(self.scores, bins=50)
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        # 绘制最大方块分布
        plt.subplot(2, 2, 2)
        tile_values = sorted(list(set(self.max_tiles)))
        tile_counts = [self.max_tiles.count(v) for v in tile_values]
        
        # 转换为百分比
        tile_percentages = [(count / num_games) * 100 for count in tile_counts]
        
        plt.bar([str(v) for v in tile_values], tile_percentages)
        plt.title('Max Tile Distribution')
        plt.xlabel('Tile Value')
        plt.ylabel('Percentage (%)')
        
        # 绘制分数累积分布
        plt.subplot(2, 2, 3)
        sorted_scores = np.sort(self.scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        plt.plot(sorted_scores, cumulative)
        plt.title('Score Cumulative Distribution')
        plt.xlabel('Score')
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        
        # 绘制分数箱线图
        plt.subplot(2, 2, 4)
        plt.boxplot(self.scores)
        plt.title('Score Box Plot')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(f"random_agent_results_{num_games}.png")
        plt.close()

def play_game_worker(seed):
    """用于多进程的游戏函数"""
    # 设置随机种子确保每个进程有不同的随机序列
    random.seed(seed)
    np.random.seed(seed)
    
    agent = RandomAgent()
    return agent.play_game(render=False, enable_recording=False)

def main():
    parser = argparse.ArgumentParser(description='2048随机策略智能体')
    parser.add_argument('--play', action='store_true', help='玩一局游戏并显示过程')
    parser.add_argument('--evaluate', action='store_true', help='评估随机策略的性能')
    parser.add_argument('--parallel', action='store_true', help='使用多进程并行评估')
    parser.add_argument('--games', type=int, default=1000, help='评估时的游戏局数')
    parser.add_argument('--processes', type=int, default=None, help='并行评估时使用的进程数')
    
    args = parser.parse_args()
    
    agent = RandomAgent()
    
    if args.play:
        print("使用随机策略玩一局游戏...")
        agent.play_game(render=True)
    
    if args.evaluate:
        if args.parallel:
            print(f"并行评估随机策略 ({args.games} 局游戏)...")
            agent.evaluate_parallel(num_games=args.games, num_processes=args.processes)
        else:
            print(f"评估随机策略 ({args.games} 局游戏)...")
            agent.evaluate(num_games=args.games)
    
    # 如果没有提供参数，则默认并行评估10万局
    if not (args.play or args.evaluate):
        # 如果指定了--parallel参数但没有--evaluate，也执行并行评估
        if args.parallel:
            print(f"并行评估随机策略 ({args.games} 局游戏)...")
            agent.evaluate_parallel(num_games=args.games, num_processes=args.processes)
        else:
            print("没有提供参数，默认并行评估随机策略 (100000 局游戏)...")
            agent.evaluate_parallel(num_games=100000)

if __name__ == "__main__":
    main() 