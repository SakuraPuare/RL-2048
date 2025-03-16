from setuptools import setup, find_packages

setup(
    name="rl-2048",
    version="0.1.0",
    description="2048 Game with Reinforcement Learning Agents",
    author="RL-2048 Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pygame",
        "torch",
        "matplotlib",
        "tqdm"
    ],
    entry_points={
        'console_scripts': [
            'rl2048=main:main',
        ],
    },
    python_requires='>=3.6',
) 