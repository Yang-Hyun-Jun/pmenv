# pmenv

Introduction to the **pmenv** Package (made by *Yang-Hyun-Jun*)

The pmenv is a reinforcement learning environment for the Portfolio Rebalancing Task. It loads multiple stock market data using tickers and synchronizes them into a 3D tensor format suitable for learning. It is an efficient environment for implementing and testing various reinforcement learning algorithms.


# Overview

The pmenv is composed of two modules, DataManager and Environment:

- The Environment module plays a role in simulating the stock market environment and handling the transitions caused by price changes and transactions. The observation is a vector of Open, High, Low, Close price and Volume and the action space is $[-1,1]^N$ where $N$ is a number of risky assets in the portfolio. Portfolio is a vector of weights of all assets and the first entry always represents cash weight.  

- The DataManager module is responsible for loading multiple stock market data using tickers and synchronizing them into a unified format suitable for learning. 

The mathematical details of the environment are implemented based on the following paper. Please refer to the paper for further information. [*An intelligent financial portfolio trading strategy using deep Q-learning, ESWA, 2020*](https://www.sciencedirect.com/science/article/pii/S0957417420303973)

# Install

    pip install pmenv

# Basic Usage
```python
import pmenv

datamanager = pmenv.DataManager()

# Tickers and Period
tickers = ["AAPL", "INCY", "REGN"]

train_start = "2019-01-02"
train_end = "2021-06-30"

# Load stock OHLCV dataframe with a ticker
data = datamanager.get_data(tickers[0], train_start, train_end)
# Load stock data tensor with ticker list
data_tensor = datamanager.get_data_tensor(tickers, train_start, train_end)


environment = pmenv.Environment(data_tensor)

# Set transaction cost
config = {"CHARGE": 0.0025, "TEX": 0.001}
environment.config = config

# Initialize the environment with balance
balance = 5000
observation = environment.reset(balance)

# Step
action = np.random.random(size=(len(tickers)))
next_observation, reward, done = environment.step(action)
```

# Customizing
```python
# Yon can define your MDP
import pmenv

# Example 
class CustomEnvironment(pmenv.Environment)
	def __init__(self):
		super().__init__()
	
	def get_state(self, observation):
		"""
		state is defined as [ohlcv, portfolio]
		"""
		portfolio = portfolio[:,np.newaxis]
		state = np.concatenate([obsevation, portfolio], axis=1)
		return state
	
	def get_reward(self):
		"""
		reward is defined as ~
		"""
```

# Tutorial with RL 

 1. Portfolio Rebalancing with Deep Q-Learning
 2. Portfolio Rebalancing with DDPG
 3. Portfolio Rebalancing with Dirichlet Distribution Trader
 
