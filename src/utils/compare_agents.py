#!/usr/bin/env python3
"""
比较不同智能体在2048游戏中的性能
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from collections import Counter
import sys
import os

# 设置matplotlib使用英文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Import from the new structure
from src.core.game import Game2048
from src.agents.random_agent import RandomAgent
from src.agents.rl_agent import DQNAgent
from src.agents.cnn_rl_agent import CNNDQNAgent

def evaluate_random_agent(num_games=100):
    """评估随机策略智能体"""
    print(f"\n正在评估随机策略智能体 ({num_games} 局游戏)...")
    agent = RandomAgent()
    return agent.evaluate(num_games=num_games)

def evaluate_dqn_agent(model_path="models/rl_model.pth", num_games=100):
    """评估DQN智能体"""
    print(f"\n正在评估DQN智能体 ({num_games} 局游戏)...")
    agent = DQNAgent()
    
    # 加载模型
    agent.load_model(model_path)
    
    scores = []
    max_tiles = []
    
    progress_bar = tqdm(range(num_games), desc="评估DQN智能体")
    
    for _ in progress_bar:
        score, max_tile = dqn_play_game(agent, model_path=model_path, render=False)
        scores.append(score)
        max_tiles.append(max_tile)
        
        # 更新进度条
        progress_bar.set_postfix({
            '分数': score,
            '最大方块': max_tile
        })
    
    # 分析结果
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    tile_counter = Counter(max_tiles)
    
    print(f"\nDQN智能体评估结果 ({num_games} 局游戏):")
    print(f"平均分数: {avg_score:.2f}")
    print(f"最高分数: {max_score}")
    print("\n最大方块分布:")
    
    # 按方块值排序
    for tile in sorted(tile_counter.keys()):
        percentage = (tile_counter[tile] / num_games) * 100
        print(f"方块 {tile}: {tile_counter[tile]} 次 ({percentage:.2f}%)")
    
    return avg_score, max_score, tile_counter, scores, max_tiles

def evaluate_cnn_agent(model_path="models/cnn_rl_model.pth", num_games=100):
    """评估CNN-DQN智能体"""
    print(f"\n正在评估CNN-DQN智能体 ({num_games} 局游戏)...")
    agent = CNNDQNAgent()
    
    # 加载模型
    agent.load_model(model_path)
    
    scores = []
    max_tiles = []
    
    progress_bar = tqdm(range(num_games), desc="评估CNN-DQN智能体")
    
    for _ in progress_bar:
        score, max_tile = cnn_play_game(agent, model_path=model_path, render=False)
        scores.append(score)
        max_tiles.append(max_tile)
        
        # 更新进度条
        progress_bar.set_postfix({
            '分数': score,
            '最大方块': max_tile
        })
    
    # 分析结果
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    tile_counter = Counter(max_tiles)
    
    print(f"\nCNN-DQN智能体评估结果 ({num_games} 局游戏):")
    print(f"平均分数: {avg_score:.2f}")
    print(f"最高分数: {max_score}")
    print("\n最大方块分布:")
    
    # 按方块值排序
    for tile in sorted(tile_counter.keys()):
        percentage = (tile_counter[tile] / num_games) * 100
        print(f"方块 {tile}: {tile_counter[tile]} 次 ({percentage:.2f}%)")
    
    return avg_score, max_score, tile_counter, scores, max_tiles

def compare_agents(games=100, dqn_model="models/rl_model.pth", cnn_model="models/cnn_rl_model.pth"):
    """比较所有智能体的性能"""
    # 评估随机策略
    random_avg_score, random_max_score, random_tile_counter = evaluate_random_agent(games)
    
    # 评估DQN智能体
    try:
        dqn_avg_score, dqn_max_score, dqn_tile_counter, dqn_scores, dqn_max_tiles = evaluate_dqn_agent(dqn_model, games)
        has_dqn = True
    except Exception as e:
        print(f"无法评估DQN智能体: {e}")
        has_dqn = False
    
    # 评估CNN-DQN智能体
    try:
        cnn_avg_score, cnn_max_score, cnn_tile_counter, cnn_scores, cnn_max_tiles = evaluate_cnn_agent(cnn_model, games)
        has_cnn = True
    except Exception as e:
        print(f"无法评估CNN-DQN智能体: {e}")
        has_cnn = False
    
    # 绘制比较图表
    plt.figure(figsize=(15, 10))
    
    # 1. 平均分数比较
    plt.subplot(2, 2, 1)
    labels = ['Random']
    values = [random_avg_score]
    
    if has_dqn:
        labels.append('DQN')
        values.append(dqn_avg_score)
    
    if has_cnn:
        labels.append('CNN-DQN')
        values.append(cnn_avg_score)
    
    plt.bar(labels, values)
    plt.title('Average Score Comparison')
    plt.ylabel('Score')
    
    # 2. 最高分数比较
    plt.subplot(2, 2, 2)
    labels = ['Random']
    values = [random_max_score]
    
    if has_dqn:
        labels.append('DQN')
        values.append(dqn_max_score)
    
    if has_cnn:
        labels.append('CNN-DQN')
        values.append(cnn_max_score)
    
    plt.bar(labels, values)
    plt.title('Max Score Comparison')
    plt.ylabel('Score')
    
    # 3. 最大方块分布比较
    plt.subplot(2, 2, 3)
    
    # 获取所有可能的方块值
    all_tiles = set(random_tile_counter.keys())
    if has_dqn:
        all_tiles.update(dqn_tile_counter.keys())
    if has_cnn:
        all_tiles.update(cnn_tile_counter.keys())
    
    all_tiles = sorted(all_tiles)
    
    # 计算每个智能体的方块分布百分比
    random_percentages = [random_tile_counter.get(tile, 0) / games * 100 for tile in all_tiles]
    
    if has_dqn:
        dqn_percentages = [dqn_tile_counter.get(tile, 0) / games * 100 for tile in all_tiles]
    
    if has_cnn:
        cnn_percentages = [cnn_tile_counter.get(tile, 0) / games * 100 for tile in all_tiles]
    
    # 设置柱状图
    x = np.arange(len(all_tiles))
    width = 0.2
    
    plt.bar(x - width, random_percentages, width, label='Random')
    
    if has_dqn:
        plt.bar(x, dqn_percentages, width, label='DQN')
    
    if has_cnn:
        plt.bar(x + width, cnn_percentages, width, label='CNN-DQN')
    
    plt.xlabel('Tile Value')
    plt.ylabel('Percentage (%)')
    plt.title('Max Tile Distribution Comparison')
    plt.xticks(x, [str(tile) for tile in all_tiles])
    plt.legend()
    
    # 4. 分数分布比较（箱线图）
    plt.subplot(2, 2, 4)
    
    data = [random_avg_score]
    labels = ['Random']
    
    if has_dqn:
        data.append(dqn_scores)
        labels.append('DQN')
    
    if has_cnn:
        data.append(cnn_scores)
        labels.append('CNN-DQN')
    
    plt.boxplot(data, labels=labels)
    plt.title('Score Distribution Comparison')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig("agents_comparison.png")
    plt.close()
    
    print("\n比较结果已保存到 agents_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='比较2048智能体')
    parser.add_argument('--games', type=int, default=100, help='每个智能体评估的游戏局数')
    parser.add_argument('--dqn-model', type=str, default="models/rl_model.pth", help='DQN模型路径')
    parser.add_argument('--cnn-model', type=str, default="models/cnn_rl_model.pth", help='CNN-DQN模型路径')
    
    args = parser.parse_args()
    
    compare_agents(
        games=args.games,
        dqn_model=args.dqn_model,
        cnn_model=args.cnn_model
    )

if __name__ == "__main__":
    main() 