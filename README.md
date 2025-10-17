# Technical Documentation: Q-Learning Implementation for Taxi Environment

## 1. Project Overview

This project implements and compares two reinforcement learning approaches on the Taxi-v3 environment from Gymnasium:
1. Tabular Q-Learning
2. Deep Q-Learning (DQN)

### 1.1 Environment Description
- **Environment**: Taxi-v3 from Gymnasium
- **State Space**: 500 discrete states
- **Action Space**: 6 possible actions
- **Reward Structure**:
  - +20 for successful passenger drop-off
  - -1 per time step (penalty for longer episodes)
  - -10 for illegal pickup/drop actions

## 2. Implementation Components

### 2.1 Q-Learning Implementation
```python
Core Components:
- State-Action Value Table (Q-table): 500×6 matrix
- Exploration Strategy: ε-greedy policy
- Learning Parameters:
  - α (learning rate)
  - ε (exploration rate)
  - γ (discount factor) = 0.9
```

### 2.2 Deep Q-Network (DQN) Architecture
```python
Neural Network Structure:
- Input Layer: 500 nodes (one-hot encoded state)
- Hidden Layer 1: 128 nodes with ReLU
- Hidden Layer 2: 128 nodes with ReLU
- Output Layer: 6 nodes (Q-values for each action)
```

## 3. Hyperparameter Study

### 3.1 Q-Learning Experiments

#### Learning Rate (α) Analysis:
- **α = 0.001**: 
  - Avg Reward: -0.090
  - Avg Steps: 262.71
  - Too slow learning
- **α = 0.01**: 
  - Avg Reward: -0.804
  - Avg Steps: 231.82
  - Moderate learning speed
- **α = 0.2**: 
  - Avg Reward: -3.414
  - Avg Steps: 0.99
  - Fast learning but potentially unstable

#### Exploration Rate (ε) Analysis:
- **ε = 0.2**:
  - Avg Reward: -3.376
  - Avg Steps: 24.24
  - Balanced exploration-exploitation
- **ε = 0.3**:
  - Avg Reward: -3.634
  - Avg Steps: 38.94
  - Higher exploration, slower convergence

### 3.2 Optimal Hyperparameters
Based on experiments, the best configuration was:
- Learning Rate (α) = 0.1
- Exploration Rate (ε) = 0.1
- Discount Factor (γ) = 0.9

## 4. Implementation Features

### 4.1 Training Utilities
1. **State Decoding**:
   - Function to decode integer state into components:
     - Taxi row position
     - Taxi column position
     - Passenger location
     - Destination

2. **Metrics Collection**:
   - Episodes completed
   - Steps per episode
   - Rewards per episode
   - Moving averages for visualization

### 4.2 Evaluation Methods
1. **Policy Evaluation**:
   - Greedy policy testing
   - 100-200 evaluation episodes
   - Performance metrics:
     - Mean return
     - Mean steps per episode

### 4.3 Visualization Tools
- Training curves:
  - Moving average rewards
  - Steps per episode
- Hyperparameter comparison plots
- Results saved to CSV files for analysis

## 5. DQN vs Q-Learning Comparison

### 5.1 Advantages/Disadvantages

#### Q-Learning:
- **Pros**:
  - Faster convergence for small state spaces
  - Simpler implementation
  - More stable learning
- **Cons**:
  - Limited scalability
  - Memory inefficient for large state spaces

#### DQN:
- **Pros**:
  - Better generalization
  - Suitable for large state spaces
  - Memory efficient
- **Cons**:
  - Longer training time
  - More complex implementation
  - Requires careful hyperparameter tuning

## 6. Project Structure
```
.
├── Q learning taxi.ipynb        # Main implementation notebook
└── results_taxi              # Results directory
    ├── alpha_*.csv            # Learning rate experiments
    ├── eps_*.csv             # Exploration rate experiments
    ├── baseline_*.csv        # Baseline results
    └── hyperparameter_summary.csv
```

## 7. Dependencies
- gymnasium
- numpy
- pandas
- matplotlib
- torch (for DQN)
- tqdm (for progress bars)

## 8. Usage Instructions

## Installation
1.	Clone this repository:
 	git clone https://github.com/SHIRU235/QLearning.git .
2.	Create a virtual environment (optional but recommended):
 	python -m venv venv
source venv/bin/activate   # On Linux/Mac
.venv\scripts\activate      # On Windows
3.	Install dependencies:
 	pip install -r requirements.txt


### 8.2 Running Experiments
1. Open the Jupyter notebook `Qlearning.ipynb`
2. Execute cells in sequence
3. Results will be saved in the `results_taxi` directory

### 8.3 Analyzing Results
- Check the generated CSV files in `results_taxi`
- View training curves and performance plots
- Compare hyperparameter configurations using summary files

## 9. Future Improvements
1. Implementation of different exploration strategies
2. Addition of prioritized experience replay for DQN
3. Integration of other RL algorithms for comparison
4. Enhanced visualization and analysis tools
5. Support for different Gymnasium environments