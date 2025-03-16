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
    
    # Determine which agent script to run
    agent_script = "rl_agent.py" if args.agent == 'dqn' else "cnn_rl_agent.py"
    
    # Set default model path if not specified
    if args.model is None:
        args.model = "rl_model.pth" if args.agent == 'dqn' else "cnn_rl_model.pth"
    
    # Build command
    cmd = [sys.executable, agent_script]
    
    if args.train:
        cmd.append("--train")
    if args.play:
        cmd.append("--play")
    
    cmd.extend(["--episodes", str(args.episodes)])
    cmd.extend(["--model", args.model])
    
    print(f"Running {args.agent.upper()} agent...")
    
    # Execute the command
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main() 