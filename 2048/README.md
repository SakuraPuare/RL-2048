# 2048 Game

A Python implementation of the popular 2048 game with both text-based and graphical interfaces.

## Game Rules

- The game is played on a 4×4 grid.
- The game starts with two tiles (value 2 or 4) already on the grid.
- Use arrow keys to move all tiles in one direction.
- When two tiles with the same number touch, they merge into one with their sum.
- After each move, a new tile (2 or 4) appears in a random empty cell.
- The game is won when a tile with the value 2048 appears.
- The game ends when there are no valid moves left.

## Requirements

- Python 3.6+
- NumPy
- Pygame (optional, for GUI mode)

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install numpy
pip install pygame  # Optional, for GUI mode
```

## How to Play

### Running the Game

```bash
# Run with GUI (default)
python main.py

# Run in text mode
python main.py --text
```

### Controls

- **GUI Mode**:
  - Arrow keys or WASD to move tiles
  - ESC to quit

- **Text Mode**:
  - W or Up Arrow: Move Up
  - A or Left Arrow: Move Left
  - S or Down Arrow: Move Down
  - D or Right Arrow: Move Right
  - Q: Quit

## Files

- `game.py`: Core game logic
- `text_interface.py`: Text-based interface for terminal play
- `gui_interface.py`: Graphical interface using Pygame
- `main.py`: Entry point that allows choosing between text and GUI modes

## Screenshots

(Add screenshots here)

## License

This project is open source and available under the MIT License. 