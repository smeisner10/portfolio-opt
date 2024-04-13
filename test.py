import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense
import tensorflow as tf
import datetime
import yfinance as yahooFinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)


tf.random.set_seed(1234)
tf.keras.utils.set_random_seed(1234)


class Model:
    def __init__(self):
        self.data = None
        self.model = None

    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio

        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])

            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)

            portfolio_returns = (
                portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

            # since we want to maximize Sharpe, while gradient descent minimizes the loss,
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe

        model.compile(loss=sharpe_loss, optimizer='adam')
        return model

    def get_allocations(self, data, epochs):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data

        input: data - DataFrame of historical closing prices of various assets

        return: the allocations ratios for each of the given assets
        '''

        # data with returns
        data_w_ret = np.concatenate(
            [data.values[1:], data.pct_change().values[1:]], axis=1)

        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)

        if self.model is None:
            self.model = self.__build_model(
                data_w_ret.shape, len(data.columns))

        fit_predict_data = data_w_ret[np.newaxis, :]
        self.model.fit(fit_predict_data, np.zeros(
            (1, len(data.columns))), epochs=epochs, shuffle=False, verbose=True)
        return self.model.predict(fit_predict_data)[0]


# can change this, but must ALSO change get_data function
labels = ["vti", "agg", "dbc", "vix"]
N_ASSETS = len(labels)


def get_data(startDate, endDate):
    vti = yahooFinance.Ticker("VTI").history(
        start=startDate, end=endDate).reset_index()
    agg = yahooFinance.Ticker("AGG").history(
        start=startDate, end=endDate).reset_index()
    dbc = yahooFinance.Ticker("DBC").history(
        start=startDate, end=endDate).reset_index()
    vix = yahooFinance.Ticker("^VIX").history(
        start=startDate, end=endDate).reset_index()
    data = pd.DataFrame()
    all_assets = [vti, agg, dbc, vix]

    for i, asset in enumerate(all_assets):
        asset = asset.reset_index()
        lb = labels[i]
        data[lb] = asset['Close']
    return data


def prep_data_for_pred(data):
    data_w_ret = np.concatenate(
        [data.values[1:], data.pct_change().values[1:]], axis=1)
    fit_predict_data = data_w_ret[np.newaxis, :]
    return fit_predict_data


money = 1
window = 365  # days

startDate = datetime.datetime(2010, 1, 1)
endDate = startDate + datetime.timedelta(days=window)

data = get_data(startDate, endDate)
model = Model()
window = data.shape[0] - 1  # TRUE WINDOW AFTER GETTING JUST TRADING DAYS
weights = model.get_allocations(data.iloc[:-1], epochs=100)

# how long we backtest for
full_data = get_data(startDate, datetime.datetime(2021, 1, 1))

moneys = []
for i in range(len(full_data) - window):
    sub_data = full_data.iloc[i:i+window - 1]
    sub_data = prep_data_for_pred(sub_data)

    weights = model.model.predict(sub_data)[0]
    returns = sub_data[0][-1] / sub_data[0][-2]
    returns = returns[:N_ASSETS]
    print(weights)
    port_returns = weights@returns
    if port_returns == port_returns:
        money *= port_returns
    startDate += datetime.timedelta(days=1)
    endDate += datetime.timedelta(days=1)
    print(i, money)
    moneys.append(money)

plt.plot(range(len(full_data) - window), moneys, label="Portfolio")
plt.xlabel("Days")
plt.yscale('log')
plt.ylabel("Returns (Log scale)")
plt.legend()
plt.show()
