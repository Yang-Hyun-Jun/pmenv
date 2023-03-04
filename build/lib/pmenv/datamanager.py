import pandas as pd
import numpy as np
import FinanceDataReader as fdr

class DataManager:
    """
    DataManager for reading and preprosseing stock datas
    """

    def get_data(self, ticker:str, start_date=None, end_date=None):
        """
        A function to load a stock dataframe from fdr.
        """
        data = fdr.DataReader(ticker, start_date, end_date)
        data = self.get_price(data)
        data.index = pd.to_datetime(data.index)
        data = data[start_date:end_date]
        return data

    def get_data_tensor(self, tickers:list, start_date=None, end_date=None):
        """
        A function to make a stock tensor data with shape (L, K, F)
        All dataframe are stacked on a common business day
        First matrix of the data tensor is one matrix for cash (risk free) asset

        L: Len of data
        K: Num of portfolio assets (except for cash)
        F: Num of features
        """

        data_tensor = []
        date = set(pd.date_range(start_date, end_date))
        
        for ticker in tickers:
            data = self.get_data(ticker, start_date, end_date)
            data_tensor.append(data)
            date = date & set(data.index.unique())

        date = sorted(date)
        data_tensor = list(map(lambda x:x[x.index.isin(date)], data_tensor))
        data_tensor = list(map(lambda x:x.to_numpy(), data_tensor))
        data_tensor = np.stack(data_tensor, axis=-1)
        data_tensor = np.concatenate([np.ones_like(data_tensor[:,:,0:1]), data_tensor], axis=-1)
        data_tensor = np.swapaxes(data_tensor, 1, 2)

        print("Data Start Date:", date[0])
        print("Data End Date:", date[-1], "\n")
        return data_tensor

    @staticmethod
    def get_price(data):
        data["Price"] = data["Close"].values
        data = data.drop(["Close"], axis=1)
        return data