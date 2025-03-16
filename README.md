# 2048 强化学习智能体

本项目实现了多种智能体来玩2048游戏，包括基于强化学习的智能体和随机策略智能体。

## 安装依赖

安装所需的依赖项：

```bash
pip install -r requirements.txt
```

## 智能体类型

本项目包含三种不同的智能体：

### 1. DQN 智能体（全连接神经网络）

使用全连接神经网络的基础DQN智能体。

```bash
python rl_agent.py [选项]
```

### 2. CNN-DQN 智能体（卷积神经网络）

使用卷积神经网络进行特征提取的高级智能体。

```bash
python cnn_rl_agent.py [选项]
```

### 3. 随机策略智能体

随机选择有效动作的基准智能体，用于评估随机策略的性能上限。

```bash
python random_agent.py [选项]
```

## 使用方法

### 统一智能体运行器

为了方便使用，您可以使用统一的智能体运行脚本：

```bash
python run_agent.py --agent [dqn|cnn] [选项]
```

示例：
```bash
# 训练CNN智能体2000回合
python run_agent.py --agent cnn --train --episodes 2000

# 使用训练好的DQN智能体玩一局游戏
python run_agent.py --agent dqn --play
```

### 随机策略智能体选项

随机策略智能体支持以下命令行选项：

```bash
# 使用随机策略玩一局游戏并显示过程
python random_agent.py --play

# 评估随机策略的性能（默认1000局游戏）
python random_agent.py --evaluate --games 1000
```

### 比较不同智能体

比较不同智能体的性能：

```bash
# 比较所有智能体（每种100局游戏）
python compare_agents.py --games 100

# 使用自定义模型路径
python compare_agents.py --dqn-model path/to/dqn_model.pth --cnn-model path/to/cnn_model.pth
```

### 强化学习智能体选项

两种强化学习智能体都支持以下命令行选项：

#### 训练智能体

训练智能体：

```bash
python rl_agent.py --train --episodes 1000
# 或
python cnn_rl_agent.py --train --episodes 1000
```

您可以使用 `--episodes` 参数指定训练回合数。

#### 使用训练好的智能体玩游戏

使用训练好的智能体玩一局游戏：

```bash
python rl_agent.py --play
# 或
python cnn_rl_agent.py --play
```

#### 训练并玩游戏

训练智能体然后玩一局游戏：

```bash
python rl_agent.py
# 或
python cnn_rl_agent.py
```

## 工作原理

### DQN 智能体

基础DQN智能体使用：
- **状态表示**：一维热编码的4x4网格
- **动作空间**：四个可能的动作（上、右、下、左）
- **奖励函数**：每次移动后的分数增加
- **神经网络**：具有三个隐藏层的全连接神经网络
- **经验回放**：存储和采样过去的经验
- **目标网络**：用于稳定Q值目标的单独网络

### CNN-DQN 智能体

基于CNN的智能体对基础DQN进行了改进：
- **状态表示**：16通道4x4网格（每个2的幂一个通道）
- **卷积层**：从网格中提取空间特征
- **增强的奖励函数**：同时奖励分数增加和达到新的最大方块
- **更好的探索**：更有效地探索状态空间

### 随机策略智能体

随机策略智能体：
- **动作选择**：在每一步随机选择一个有效动作
- **基准性能**：用于评估其他智能体相对于随机策略的性能提升
- **性能上限**：揭示随机策略在2048游戏中能达到的最大方块和分数

## 训练进度

在训练过程中，会跟踪并可视化智能体的进度：
- 每回合的分数
- 每回合达到的最大方块
- 探索率（epsilon）的变化

训练进度图表会保存为PNG文件。

## 模型保存/加载

训练好的模型会保存到：
- 基础DQN智能体：`rl_model.pth`
- CNN-DQN智能体：`cnn_rl_model.pth`

您可以使用 `--model` 参数指定不同的路径。

## 性能比较

使用 `compare_agents.py` 脚本可以比较不同智能体的性能，包括：
- 平均分数比较
- 最高分数比较
- 最大方块分布比较
- 分数分布比较

比较结果会保存为 `agents_comparison.png`。 