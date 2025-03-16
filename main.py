#!/usr/bin/env python3
"""
2048 Game with Reinforcement Learning Agents
Main entry point for the application
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='2048 Game with RL Agents')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Play game command
    play_parser = subparsers.add_parser('play', help='Play 2048 game')
    play_parser.add_argument('--interface', choices=['gui', 'text'], default='gui',
                          help='Interface to use (default: gui)')
    
    # Train agent command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--agent', choices=['dqn', 'cnn', 'random'], default='dqn',
                           help='Agent type to train (default: dqn)')
    train_parser.add_argument('--episodes', type=int, default=1000,
                           help='Number of episodes to train (default: 1000)')
    train_parser.add_argument('--model-path', type=str, default=None,
                           help='Path to save the trained model')
    
    # Evaluate agent command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate an agent')
    eval_parser.add_argument('--agent', choices=['dqn', 'cnn', 'random'], default='dqn',
                          help='Agent type to evaluate (default: dqn)')
    eval_parser.add_argument('--games', type=int, default=100,
                          help='Number of games to evaluate (default: 100)')
    eval_parser.add_argument('--model-path', type=str, default=None,
                          help='Path to the model to evaluate')
    
    # Compare agents command
    compare_parser = subparsers.add_parser('compare', help='Compare different agents')
    compare_parser.add_argument('--games', type=int, default=100,
                             help='Number of games for each agent (default: 100)')
    compare_parser.add_argument('--dqn-model', type=str, default='models/rl_model.pth',
                             help='Path to DQN model')
    compare_parser.add_argument('--cnn-model', type=str, default='models/cnn_rl_model.pth',
                             help='Path to CNN model')
    
    args = parser.parse_args()
    
    if args.command == 'play':
        from src.core.main import main as play_game
        play_game(args.interface)
    
    elif args.command == 'train':
        if args.agent == 'dqn':
            from src.agents.rl_agent import train_agent
            model_path = args.model_path or 'models/rl_model.pth'
            train_agent(episodes=args.episodes, model_path=model_path)
        
        elif args.agent == 'cnn':
            from src.agents.cnn_rl_agent import train_agent
            model_path = args.model_path or 'models/cnn_rl_model.pth'
            train_agent(episodes=args.episodes, model_path=model_path)
        
        elif args.agent == 'random':
            from src.agents.random_agent import evaluate_random_agent
            evaluate_random_agent(games=args.episodes)
    
    elif args.command == 'evaluate':
        if args.agent == 'dqn':
            from src.agents.rl_agent import play_game
            model_path = args.model_path or 'models/rl_model.pth'
            play_game(None, model_path=model_path, render=False)
        
        elif args.agent == 'cnn':
            from src.agents.cnn_rl_agent import play_game
            model_path = args.model_path or 'models/cnn_rl_model.pth'
            play_game(None, model_path=model_path, render=False)
        
        elif args.agent == 'random':
            from src.agents.random_agent import evaluate_random_agent
            evaluate_random_agent(games=args.games)
    
    elif args.command == 'compare':
        from src.utils.compare_agents import compare_agents
        compare_agents(games=args.games, dqn_model=args.dqn_model, cnn_model=args.cnn_model)
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 