#!/usr/bin/env python3
"""
Script to run either the basic DQN or CNN-DQN agent for 2048.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run 2048 RL Agent')
    parser.add_argument('--agent', type=str, choices=['dqn', 'cnn'], default='dqn',
                        help='Agent type: dqn (basic) or cnn (advanced)')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--play', action='store_true', help='Play a game with the trained agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--model', type=str, help='Path to save/load the model')
    
    args = parser.parse_args()
    
    # Set default model path if not specified
    if args.model is None:
        args.model = "models/rl_model.pth" if args.agent == 'dqn' else "models/cnn_rl_model.pth"
    
    if args.agent == 'dqn':
        if args.train and args.play:
            from src.agents.rl_agent import train_agent, play_game
            train_agent(episodes=args.episodes, model_path=args.model)
            play_game(None, model_path=args.model)
        elif args.train:
            from src.agents.rl_agent import train_agent
            train_agent(episodes=args.episodes, model_path=args.model)
        elif args.play:
            from src.agents.rl_agent import play_game
            play_game(None, model_path=args.model)
        else:
            print("Please specify --train or --play")
    else:  # CNN agent
        if args.train and args.play:
            from src.agents.cnn_rl_agent import train_agent, play_game
            train_agent(episodes=args.episodes, model_path=args.model)
            play_game(None, model_path=args.model)
        elif args.train:
            from src.agents.cnn_rl_agent import train_agent
            train_agent(episodes=args.episodes, model_path=args.model)
        elif args.play:
            from src.agents.cnn_rl_agent import play_game
            play_game(None, model_path=args.model)
        else:
            print("Please specify --train or --play")
    
    print(f"Finished running {args.agent.upper()} agent.")

if __name__ == "__main__":
    main() 