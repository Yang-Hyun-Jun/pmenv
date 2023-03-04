import numpy as np

class Environment:
    """
    Environment for Rebalancing Task 
    """
    
    config = {"CHARGE": 0.0, "TEX":0.0}

    def __init__(self, stock_tensor=None):
        """
        L: Len of data
        K: Num of portfolio assets
        F: Num of features

        stock_tensor: tensor with shape (L, K, F)
        observation: feature matrix with shape (K, F) 
        portfolio_value: portfolio valuation 
        cum_fee: cummulative transaction fee
        num_stocks: holding shares
        """
        
        self.idx = 0
        self.price_column = -1
        self.stock_tensor = stock_tensor

        self.observation = None
        self.portfolio_value = None
        self.initial_balance = None
        self.balance = None
        self.profitloss = None
        self.cum_fee = None
        self.num_stocks = None
        self.portfolio = None
        self.num_stocks = None
        self.portfolio = None

        self.K = stock_tensor.shape[1]
        self.F = stock_tensor.shape[2]-1
        self.is_reset = False

    def reset(self, balance):
        """
        initialize the environment &
        return initial state
        """
        self.is_reset = True
        self.observation = None
        self.idx = 0
        
        self.portfolio_value = balance
        self.initial_balance = balance
        self.balance = balance
        self.profitloss = 0.0
        self.cum_fee = 0.0

        self.num_stocks = np.zeros(self.K-1)
        self.portfolio = np.zeros(self.K)
        self.portfolio[0] = 1.0

        observation = self.observe()
        return observation

    def get_trading_unit(self, action, price):
        """
        A function that converts action to number of shares based on current price
        """
        trading_amount = self.portfolio_value * abs(action)
        trading_unit = trading_amount / price
        trading_unit = trading_unit.astype(int)
        return trading_unit

    def pi_operator(self, change_rate):
        """
        Auxiliary functions for calculating portfolios based on price changes
        """
        pi_vector = np.zeros(len(change_rate) + 1)
        pi_vector[0] = 1
        pi_vector[1:] = change_rate + 1
        return pi_vector

    def get_portfolio_value(self, price_now, price_next):
        """
        A function to calculate portfolio value by price change 
        """
        change_rate = (price_next - price_now) / price_now
        pi_vector = self.pi_operator(change_rate)
        portfolio_value = np.dot(self.portfolio_value * self.portfolio, pi_vector)
        return portfolio_value

    def get_portfolio(self, price_now, price_next):
        """
        A function to calculate portfolio by price change 
        """
        change_rate = (price_next - price_now) / price_now
        pi_vector = self.pi_operator(change_rate)
        portfolio = (self.portfolio * pi_vector) / (np.dot(self.portfolio, pi_vector))
        return portfolio

    def get_reward(self):
        """
        A function that returns log profitloss reward
        """
        reward = np.log(self.portfolio_value) - np.log(self.initial_balance)
        return reward

    def get_profitloss(self, pv):
        """
        A function that returns profitloss based on current portfolio value
        """
        profitloss = ((pv / self.initial_balance) -1) * 100
        return profitloss

    def step(self, action):
        """
        A function that returns the next_state, reward, and done after transactions.
        Transaction execution follows the following rules.

        1. sell order
        (1.1) Depending on the action, the order units is determined.
        (1.2) If the holding shares is not enough, it is modified to the maximum shares that can be sold.

        2. buy order
        (2.1) Depending on the action, the order units is determined.
        (2.2) Calculate the needed balances for ordering each item according to the action.
        (2.3) Split the current balance according to the ratio of needed balance of each item. 
        (2.4) Determines the optimized order untis that can be executed within the weighted_balance of each item.
        (2.5) Determines min[order units, optimized order untis] as a final order.
        """ 
        
        if not self.is_reset:
            raise Warning("Reset is needed")
        
        fee = 0 
        CHARGE = self.config["CHARGE"]
        TEX = self.config["TEX"]

        price_now = self.get_price()
        trading_units = self.get_trading_unit(action, price_now)

        # Sell Transaction Cost & Sell Stock Index
        sell_cost = CHARGE + TEX
        sell_ind = np.where( (-1<=action) & (action<0) & (trading_units>0) )[0]

        # Buy Transaction Cost & Buy Stock Index
        buy_cost = CHARGE
        buy_ind = np.where( (0<action) & (action<=1) & (trading_units>0) )[0]

        # Execute Sell Action
        trading_units[sell_ind] = np.min([trading_units[sell_ind], self.num_stocks[sell_ind]], axis=0)
        invest_amounts = price_now[sell_ind] * trading_units[sell_ind]
        fee += sum(invest_amounts) * sell_cost

        self.cum_fee += fee
        self.num_stocks[sell_ind] -= trading_units[sell_ind]
        self.balance += sum(invest_amounts) * (1-sell_cost)
        self.portfolio[0] += sum(invest_amounts) * (1-sell_cost) / self.portfolio_value
        self.portfolio[1 + sell_ind] -= invest_amounts / self.portfolio_value

        # Execute Buy Action
        needed_balances = price_now[buy_ind] * trading_units[buy_ind] * (1+buy_cost)
        weighted_balances = self.balance * (needed_balances / np.sum(needed_balances))
        optimized_units = (weighted_balances / (price_now[buy_ind] * (1+buy_cost))).astype(int)

        trading_units[buy_ind] = np.min([trading_units[buy_ind], optimized_units], axis=0)
        invest_amounts = price_now[buy_ind] * trading_units[buy_ind]
        fee += sum(invest_amounts) * sell_cost

        self.cum_fee += fee
        self.num_stocks[buy_ind] += trading_units[buy_ind]
        self.balance -= sum(invest_amounts) * (1+buy_cost)
        self.portfolio[0] -= sum(invest_amounts) * (1+buy_cost) / self.portfolio_value
        self.portfolio[1 + buy_ind] += invest_amounts / self.portfolio_value

        self.portfolio_value -= fee
        self.portfolio = self.portfolio / np.sum(self.portfolio)

        next_observation = self.observe()
        price_next = self.get_price()

        # Calculate Portfolio, PV by price change
        self.portfolio_value = self.get_portfolio_value(price_now, price_next)
        self.portfolio = self.get_portfolio(price_now, price_next)    
        self.profitloss = self.get_profitloss(self.portfolio_value)        

        # Reward, Done
        reward = np.array([self.get_reward()])
        done = np.array([1]) if len(self.stock_tensor)-1 <= self.idx else np.array([0])
        return next_observation, reward, done
    
    def observe(self):
        """
        A function to observe next features 
        """
        self.observation = self.stock_tensor[self.idx]
        self.idx += 1
        return self.observation[:,:self.price_column]

    def get_price(self):
        """
        A function to get a risky assets' price vector
        """
        return self.observation[1:,self.price_column]