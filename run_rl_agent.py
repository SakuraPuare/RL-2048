#!/usr/bin/env python3
"""
Simple script to run the 2048 RL agent.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run 2048 RL Agent')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--play', action='store_true', help='Play a game with the trained agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--model', type=str, default="models/rl_model.pth", help='Path to save/load the model')
    
    args = parser.parse_args()
    
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
    
    print("Finished running RL agent.")

if __name__ == "__main__":
    main() 