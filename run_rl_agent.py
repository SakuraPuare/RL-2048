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
    parser.add_argument('--model', type=str, default="rl_model.pth", help='Path to save/load the model')
    
    args = parser.parse_args()
    
    # Run the RL agent
    cmd = [sys.executable, "rl_agent.py"]
    
    if args.train:
        cmd.append("--train")
    if args.play:
        cmd.append("--play")
    if args.episodes != 1000:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.model != "rl_model.pth":
        cmd.extend(["--model", args.model])
    
    # Execute the command
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main() 