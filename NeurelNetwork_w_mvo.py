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
    tf.keras.utils.set_random_seed(seed)


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
            LSTM(64, input_shape=input_shape),  # LSTM layer
            Flatten(),  # matrix -> vec
            # dense layer with softmax activation
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):  # _ is bc no need for y_true.
            # y_pred = weights

            # make time-series start at 1
            data = tf.divide(self.data, self.data[0])

            # portfolio = data * weights
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)

            # sequential returns = divide by previous row
            portfolio_returns = (
                portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            # compute Sharpe
            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

            # we will minimize -Sharpe
            return -sharpe

        model.compile(loss=sharpe_loss, optimizer='adam')
        return model

    def get_allocations(self, data, epochs):
        '''
        Sets up data and trains model, returns output for predicted weights.

        input: data - DataFrame of historical closing prices of various assets

        return: the allocations ratios for each of the given assets
        '''

        # data <- returns data and price data
        data_w_ret = np.concatenate(
            [data.values[1:], data.pct_change().values[1:]], axis=1)

        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)

        if self.model is None:
            self.model = self.__build_model(
                data_w_ret.shape, len(data.columns))

        # training happens here
        fit_predict_data = data_w_ret[np.newaxis, :]
        history = self.model.fit(fit_predict_data, np.zeros(
            (1, len(data.columns))), epochs=epochs, shuffle=False, verbose=True)
        self.losses = history.history['loss']  # store losses over time
        # return next prediction
        return self.model.predict(fit_predict_data)[0]


# can change this, but must ALSO change get_data function
N_ASSETS = 4  # len(labels)


def get_data(startDate, endDate, tickers):
    """
    Get yahoo finance data from startDate to endDate for tickers in list tickers.
    Return the df.
    """
    s1 = yahooFinance.Ticker(tickers[0]).history(  # fetch each asset sequentially
        start=startDate, end=endDate).reset_index()
    s2 = yahooFinance.Ticker(tickers[1]).history(
        start=startDate, end=endDate).reset_index()
    s3 = yahooFinance.Ticker(tickers[2]).history(
        start=startDate, end=endDate).reset_index()
    s4 = yahooFinance.Ticker(tickers[3]).history(
        start=startDate, end=endDate).reset_index()
    data = pd.DataFrame()
    all_assets = [s1, s2, s3, s4]

    for i, asset in enumerate(all_assets):  # conver to df w tickers as titles
        asset = asset.reset_index()
        lb = tickers[i].lower()
        data[lb] = asset['Close']
    return data


def prep_data_for_pred(data):  # data cleaning stuff
    data_w_ret = np.concatenate(
        [data.values[1:], data.pct_change().values[1:]], axis=1)  # add returns to price data
    fit_predict_data = data_w_ret[np.newaxis, :]
    return fit_predict_data


def mean_variance_optimization(returns_df):
    '''
    Mean variance approach - dont set any target thresholds, 
    instead try to arrive at the portfolio tangent to the mean-variance frontier
    that minimizes covariance given mean returns from the previous period. 
    Requires 4d input.
    Returns MVO weights.
    '''
    returns = returns_df.mean()
    covariance = returns_df.cov()

    P = matrix(covariance.values)  # covariance matrix
    q = matrix(-returns)  # Negative identity matrix for inequality constraints
    G = matrix(-np.identity(4))  # identity
    h = matrix(np.zeros(4))  # Constraint: weights must be non-negative
    # Constraint: sum of weights must equal 1, also initializes 4 outputs
    A = matrix(1.0, (1, 4))
    b = matrix(1.0)

    solvers.options['show_progress'] = False

    # convex optimization solver. Minimize both covariance and negative expected returns.
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


def backtest(tickers, startDate, epochs, window=365, retrain_every=0, min_acceptable_loss=-0.07, backtestingEnd=datetime.datetime(2021, 1, 1)):
    """
    Get data and backtest.
    tickers: list of stock tickers from yahoo (str)
    startDate: datetime object for training start. 
    epochs: epochs to train for
    window: train and predict window size
    retrain_every: number of days before retrain. set to 0 for no retrain.
    min_acceptable_loss: if (re)training did not achieve this loss, try again (set to 1 to ignore)
    backtestingEnd: last day to stop backtesting
    """
    original_start = startDate
    money = 1  # keep track of portfolio size over time

    endDate = startDate + datetime.timedelta(days=window)
    data = get_data(startDate, endDate, tickers)  # just training data here

    model = Model()
    window = data.shape[0] - 1  # TRUE WINDOW AFTER GETTING JUST TRADING DAYS
    go = True  # control var
    while go:
        model = Model()
        weights = model.get_allocations(data.iloc[:-1], epochs=epochs)
        if model.losses[-1] < min_acceptable_loss:  # retrain until here
            go = False
    history = model.losses  # for plotting loss convergence

    # backtesting data
    full_data = get_data(startDate, backtestingEnd, tickers)

    moneys = [money]  # money history
    port_weights_time = []  # weights over time

    for i in range(len(full_data) - window):  # loop through backtest
        sub_data = full_data.iloc[i:i+window - 1]  # sliding window
        weights = None
        if retrain_every > 0 and (i + 1) % retrain_every == 0:  # if time to retrain
            go = True
            while go:
                print('retrain')
                model = Model()
                tf.keras.utils.set_random_seed(np.random.randint(1, 1000))
                weights = model.get_allocations(
                    sub_data, epochs=epochs)  # train and get weights
                if model.losses[-1] < min_acceptable_loss:
                    go = False
                    sub_data = prep_data_for_pred(sub_data)

        else:
            sub_data = prep_data_for_pred(sub_data)
            weights = model.model.predict(sub_data)[0]  # get predictions

        returns = sub_data[0][-1] / sub_data[0][-2]  # get asset returns
        returns = returns[:N_ASSETS]
        print(weights)
        port_returns = weights@returns  # get portfolio returns
        if port_returns == port_returns:
            money *= port_returns
        startDate += datetime.timedelta(days=1)  # increment window
        endDate += datetime.timedelta(days=1)
        print(i, money)
        moneys.append(money)
        port_weights_time.append(weights)

    # mvo code from here
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

    ### PLOT 0: Loss convergence ###
    plt.plot(range(epochs), history, label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss (-Sharpe) Convergence over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")
    plt.show()

    ### PLOT 1: Asset Performance ###
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

    ### PLOT 2: LSTM vs Assets####
    plt.figure()

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

    ### PLOT 3: LSTM v MVO###
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

    ### PLOT 4: LSTM weights over time ###
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

    ### PLOT 5: MVO weights ###
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

    # create and save dict of final sharpe ratios
    sharpe_dict = {}
    sharpe_dict['LSTM'] = get_sharpe(moneys)
    sharpe_dict['MVO'] = get_sharpe(money_arr[window:])

    for i, t in enumerate(tickers):
        closes = full_data.iloc[window:, i]
        sharpe_dict[t] = get_sharpe(closes)

    with open('dict.csv', 'w') as csv_file:  # save dict
        writer = csv.writer(csv_file)
        for key, value in sharpe_dict.items():
            writer.writerow([key, value])

    return money, port_weights_time, sharpe_dict


if __name__ == "__main__":
    # Define set of tickers, random seed if desired, strat and end date, and
    # hyperparameters to create an experiment.

    # original test
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
    # port_returns, port_weights, _ = backtest(
    #     tickers5, startDate, epochs=200, window=365, retrain_every=730, min_acceptable_loss=0)

    # HALF METALS HALF OILS OF TOP PERFORMERS
    tickers6 = ['BNO', 'PHYS', 'LIT', 'REMX']
    # port_returns, port_weights, _ = backtest(
    #  tickers6, startDate, epochs=500, window=365)

    tickers7 = ['GE', 'MSFT', 'TSLA', 'NVDA']
    startDate = datetime.datetime(2015, 1, 1)
    set_seed(12)
    # port_returns, port_weights, _ = backtest(
    #     tickers7, startDate, window=365, epochs=100, retrain_every=0)

    tickers8 = ['PLTM', 'BNO', 'DBA', 'GLD']
    startDate = datetime.datetime(2018, 2, 9)  # 2019 1 1
    set_seed(1234)
    # port_returns, port_weights, _ = backtest(
    #     tickers8, startDate, window=50, epochs=500, retrain_every=1, min_acceptable_loss=1, backtestingEnd=datetime.datetime(2020, 6, 1))

    # backtest(tickers8, startDate, window=50, epochs=500,
    #          retrain_every=200, min_acceptable_loss=0, backtestingEnd=datetime.datetime(2023, 1, 1))

    # tickers9 = ['GLD', 'USO', 'DBA', 'DBB']
    # set_seed(123456)
    # # port_returns, port_weights, _ = backtest(
    # #     tickers8, startDate, window=365, epochs=500, retrain_every=730, min_acceptable_loss=0, backtestingEnd=datetime.datetime(2023, 1, 1))
    # startDate = datetime.datetime(2018, 1, 1)  # 2019 1 1
    # backtest(tickers9, startDate, window=50, epochs=500,
    #          retrain_every=270, min_acceptable_loss=1, backtestingEnd=datetime.datetime(2023, 1, 1))

    tickers4 = ["USCI", "^BCOM", "GCC", "FAAR"]
    startDate = datetime.datetime(2017, 1, 1)
    port_returns, port_weights, _ = backtest(
        tickers4, startDate, epochs=200, window=50, retrain_every=1, min_acceptable_loss=1, backtestingEnd=datetime.datetime(2018, 1, 1))


# credit to https://github.com/shilewenuw for starter code
