import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Lambda
import tensorflow as tf
import datetime
import yfinance as yahooFinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import csv


def set_seed(seed):
    # for replicability
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)  # change this


class Model:
    def __init__(self):
        self.data = None  # store price data
        self.model = None  # store model weights
        self.losses = None  # store history of losses

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
        history = self.model.fit(fit_predict_data, np.zeros(
            (1, len(data.columns))), epochs=epochs, shuffle=False, verbose=True)
        self.losses = history.history['loss']
        return self.model.predict(fit_predict_data)[0]


# can change this, but must ALSO change get_data function
# labels = ["vti", "agg", "dbc", "vix"]
N_ASSETS = 4  # len(labels)


def get_data(startDate, endDate, tickers):
    s1 = yahooFinance.Ticker(tickers[0]).history(
        start=startDate, end=endDate).reset_index()
    s2 = yahooFinance.Ticker(tickers[1]).history(
        start=startDate, end=endDate).reset_index()
    s3 = yahooFinance.Ticker(tickers[2]).history(
        start=startDate, end=endDate).reset_index()
    s4 = yahooFinance.Ticker(tickers[3]).history(
        start=startDate, end=endDate).reset_index()
    data = pd.DataFrame()
    all_assets = [s1, s2, s3, s4]

    for i, asset in enumerate(all_assets):
        asset = asset.reset_index()
        lb = tickers[i].lower()
        data[lb] = asset['Close']
    return data


def prep_data_for_pred(data):
    data_w_ret = np.concatenate(
        [data.values[1:], data.pct_change().values[1:]], axis=1)
    fit_predict_data = data_w_ret[np.newaxis, :]
    return fit_predict_data


def mean_variance_optimization(returns_df):

    returns = returns_df.mean()
    covariance = returns_df.cov()

    P = matrix(covariance.values)
    q = matrix(-returns)
    # Negative identity matrix for inequality constraints
    G = matrix(-np.identity(4))
    h = matrix(np.zeros(4))  # Constraint: weights must be non-negative
    A = matrix(1.0, (1, 4))  # Constraint: sum of weights must equal 1
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)

    weights = np.array(solution['x']).flatten()
    return weights


def get_sharpe(prices):
    """
    Returns Sharpe Ratio of prices, a list of sequential stock prices
    or sizes of portfolio.
    """
    returns = np.diff(prices) / prices[:-1]

    mn = np.mean(returns)
    std = np.std(returns)

    return mn / std


def backtest(tickers, startDate, epochs, window=365, retrain_every=0, min_acceptable_loss=-0.07):
    """
    Get data and backtest.
    tickers: list of stock tickers from yahoo (str)
    startDate: datetime object for training start. 
    epochs: epochs to train for
    window: train and predict window size
    retrain_every: number of days before retrain. set to 0 for no retrain.
    """
    original_start = startDate
    money = 1

    endDate = startDate + datetime.timedelta(days=window)
    data = get_data(startDate, endDate, tickers)

    model = Model()
    window = data.shape[0] - 1  # TRUE WINDOW AFTER GETTING JUST TRADING DAYS
    go = True
    while go:
        model = Model()
        weights = model.get_allocations(data.iloc[:-1], epochs=epochs)
        if model.losses[-1] < min_acceptable_loss:
            go = False
    history = model.losses
    # how long we backtest for
    full_data = get_data(startDate, datetime.datetime(2021, 1, 1), tickers)

    moneys = [money]
    port_weights_time = []
    for i in range(len(full_data) - window):
        sub_data = full_data.iloc[i:i+window - 1]
        weights = None
        if retrain_every > 0 and (i + 1) % retrain_every == 0:
            go = True
            while go:
                print('retrain')
                model = Model()
                weights = model.get_allocations(sub_data, epochs=epochs)
                if model.losses[-1] < min_acceptable_loss:
                    go = False
                    sub_data = prep_data_for_pred(sub_data)

        else:
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
        port_weights_time.append(weights)

    # mvo from here
    prices = full_data
    windows = prices.rolling(50)
    weights_df = pd.DataFrame()

    for i in windows:
        try:
            weights = mean_variance_optimization(i)
            weights_df = pd.concat(
                [weights_df, pd.DataFrame([weights])], ignore_index=True)
        except:
            pass

    weighted_prices = pd.DataFrame(weights_df[:-1].values*prices[2:].values,
                                   columns=prices[2:].columns)

    total_daily_prices = weighted_prices.sum(axis=1)
    total_returns = total_daily_prices.pct_change(1)

    money = 1
    money_arr = [money]
    for i in total_returns[1:]:
        money = money + money * i
        money_arr.append(money)

    ### PLOT 0 ###
    plt.plot(range(epochs), history, label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss (-Sharpe) Convergence over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")
    plt.show()

    ### PLOT 1 ###
    plt.figure()
    colors = ['lightblue', 'blue', 'navy', 'purple']
    for i, t in enumerate(tickers):
        init_val = full_data[t.lower()].iloc[window]
        y = full_data[t.lower()].iloc[window:] / init_val
        plt.plot(range(len(y)), y, label=t, color=colors[i])

    # Format the datetime object into month, day, and year format
    original_start = original_start.strftime("%m-%d-%Y")

    plt.xlabel(f"Days from {original_start}")
    plt.ylabel("Returns on Assets")
    # plt.title("LSTM vs MVO Portfolio Performance\nOver Time")
    plt.legend()
    plt.grid()

    # plt.savefig('INSERT_FIG_NAME.png')
    plt.show()

    ### PLOT 2 ####
    plt.figure()

    # mvo
    # plt.plot(range(len(prices[2:])-window),
    # money_arr[0:(len(money_arr) - window)], label="MVO portfolio")
    for i, t in enumerate(tickers):
        init_val = full_data[t.lower()].iloc[window]
        y = full_data[t.lower()].iloc[window:] / init_val
        plt.plot(range(len(y)), y, label=t, color=colors[i])

    plt.plot(range(len(full_data) - window + 1),
             moneys, label="LSTM Portfolio", color='green')

    plt.xlabel(f"Days from {original_start}")
    plt.ylabel("Normalized Returns")
    # plt.title("LSTM vs MVO Portfolio Performance\nOver Time")
    plt.title("LSTM Portfolio vs Assets")
    plt.legend()
    plt.grid()

    # plt.savefig('INSERT_FIG_NAME.png')
    plt.show()

    ### PLOT 3###
    plt.figure()
    plt.plot(range(len(full_data) - window + 1),
             moneys, label="LSTM Portfolio", color='blue')
    # mvo
    plt.plot(range(len(prices[2:])-window),
             money_arr[0:(len(money_arr) - window)], label="MVO Portfolio", color='lightblue')

    plt.xlabel(f"Days from {original_start}")
    plt.ylabel("Portfolio Returns")
    plt.title("LSTM vs MVO Portfolio Performance\nOver Time")
    plt.legend()
    plt.grid()

    # plt.savefig('INSERT_FIG_NAME.png')
    plt.show()

    ### PLOT 3 ###
    plt.figure()
    # plot weights over time
    port_weights_time = np.array(port_weights_time)
    for i in range(port_weights_time.shape[1]):
        plt.plot(port_weights_time[:, i], label=tickers[i], color=colors[i])

    plt.xlabel(f"Days from {original_start}")
    plt.ylabel("LSTM Portfolio Weight")
    plt.title("LSTM Portfolio Weights over Time")
    plt.grid()
    plt.legend()

    # plt.savefig('INSERT_FIG_NAME.png')
    plt.show()

    ### PLOT 4 ###
    # mvo weights plots:
    plt.figure()
    mvo_weights_plot_df = weights_df[0:(len(money_arr) - window)]

    counter = 0
    for i in mvo_weights_plot_df.columns:
        plt.plot(mvo_weights_plot_df[i], label=prices.columns[counter])
        counter = counter + 1

    plt.xlabel(f"Days from {original_start}")
    plt.ylabel("MVO Portfolio Weight")
    plt.title("MVO Portfolio Weight Over Time")
    plt.grid()
    plt.legend()

    plt.show()

    sharpe_dict = {}
    sharpe_dict['LSTM'] = get_sharpe(moneys)
    sharpe_dict['MVO'] = get_sharpe(money_arr[window:])

    for i, t in enumerate(tickers):
        closes = full_data.iloc[window:, i]
        sharpe_dict[t] = get_sharpe(closes)

    with open('dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sharpe_dict.items():
            writer.writerow([key, value])

    return money, port_weights_time, sharpe_dict


if __name__ == "__main__":
    tickers1 = ["VTI", "AGG", "DBC", "^VIX"]
    set_seed(12345678)  # 1234
    startDate = datetime.datetime(2010, 1, 1)
    # port_returns, port_weights, _ = backtest(
    #     tickers1, startDate, window=365, epochs=300, retrain_every=0)

    # sugar, corn future, soybean future, wheat
    tickers2 = ["SB=F", "KC=F", "SOYB", "WEAT"]
    startDate = datetime.datetime(2013, 1, 1)
    # port_returns, port_weights, _ = backtest(tickers2, startDate, epochs=1000)

    # BROAD COMMODITIES (nothing specific)
    tickers3 = ["USCI", "^BCOM", "GCC", "FAAR"]
    startDate = datetime.datetime(2017, 1, 1)
    # port_returns, port_weights, _ = backtest(
    #     tickers3, startDate, window=365, epochs=200, retrain_every=730, min_acceptable_loss=-.02)

    # PRECIOUS METALS AND OIL
    tickers4 = ['GLTR', 'BNO', 'PHYS', 'PSLV']
    startDate = datetime.datetime(2017, 1, 1)
    # port_returns, port_weights, _ = backtest(
    #     tickers4, startDate, epochs=200, retrain_every=730, min_acceptable_loss=-.07)

    # STRATEGIC MINERALS AND RESOURCES
    tickers5 = ['COPX', 'URA', 'LIT', 'REMX']
    startDate = datetime.datetime(2017, 1, 1)
    port_returns, port_weights, _ = backtest(
        tickers5, startDate, epochs=200, window=365, retrain_every=730, min_acceptable_loss=0)

    # HALF METALS HALF OILS OF TOP PERFORMERS
    tickers6 = ['BNO', 'PHYS', 'LIT', 'REMX']
    # port_returns, port_weights, _ = backtest(
    #  tickers6, startDate, epochs=500, window=365)

    tickers7 = ['GE', 'MSFT', 'TSLA', 'NVDA']
    startDate = datetime.datetime(2015, 1, 1)
    set_seed(12)
    # port_returns, port_weights, _ = backtest(
    #     tickers7, startDate, window=365, epochs=100, retrain_every=0)
