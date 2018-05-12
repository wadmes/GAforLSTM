import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def one_day_simulation(prediction, upper_threshold, lower_threshold, stock_price, share_num, money, sentiment_weight, adjust_ratio):
    if prediction>=upper_threshold:
        if sentiment_weight > 0:
            buy_ratio = min((prediction - upper_threshold + adjust_ratio), 1.0)
            buy_ratio = max(buy_ratio, 0)
        elif sentiment_weight < 0:
            buy_ratio = min((prediction - upper_threshold - adjust_ratio), 1.0)
            buy_ratio = max(buy_ratio, 0)
        else:
            buy_ratio = min((prediction - upper_threshold), 1.0)
            buy_ratio = max(buy_ratio, 0)
        share_num += int(money * buy_ratio / stock_price)
        money -= int(money * buy_ratio / stock_price) * stock_price
    elif prediction<=lower_threshold:
        if sentiment_weight > 0:
            sell_ratio = min((prediction - upper_threshold - adjust_ratio), 1.0)
            sell_ratio = max(sell_ratio, 0)
        elif sentiment_weight < 0:
            sell_ratio = min((prediction - upper_threshold + adjust_ratio), 1.0)
            sell_ratio = max(sell_ratio, 0)
        else:
            sell_ratio = min((prediction - upper_threshold), 1.0)
            sell_ratio = max(sell_ratio, 0)
        share_num -= int(share_num * sell_ratio)
        money += int(share_num * sell_ratio) * stock_price
    return share_num, money


def simulation_algorithm(sentiment, lstm_prediction, real, upper_threshold, 
    lower_threshold, adjust_range, adjust_ratio):
    money = 10000.0
    share_num = 0
    capital_simulate = []
    capital_stock = []
    gap = len(real) - lstm_prediction.shape[0] # gap >= 1
    stock_price=real[gap-1]
    date_length = lstm_prediction.shape[0]
    sentiment = sentiment.iloc[-(date_length+1):-1]
    mean_sentiment = sentiment['Average'].mean()
    news_number = sentiment['Total'].mean()
    ix = 0
    for index, data in sentiment.iterrows():
        if data['Total']>news_number and data['Average']>mean_sentiment+adjust_range:
            sentiment_weight = 1
        elif data['Total']>news_number and data['Average']<mean_sentiment-adjust_range:
            sentiment_weight = -1
        else:
            sentiment_weight = 0
        share_num, money = one_day_simulation(lstm_prediction[ix], upper_threshold, lower_threshold, stock_price, share_num, money, sentiment_weight, adjust_ratio)
        stock_price = real[gap+ix]
        capital_simulate.append(money + stock_price * share_num)
        capital_stock.append(stock_price * int(10000/real[gap-1]))
        ix += 1
    return capital_simulate, capital_stock

def random_guess(real):
    money = 10000.0
    share_num = 0
    capital_simulate = []
    capital_stock = []
    stock_price=real[0]
    for i in range(1,len(real)):
        if random.uniform(0,1) < 0.5:
            buy_ratio = random.uniform(0,1)
            share_num += int(money * buy_ratio / stock_price)
            money -= int(money * buy_ratio / stock_price) * stock_price
        else:
            sell_ratio = random.uniform(0,1)
            share_num -= int(share_num * sell_ratio)
            money += int(share_num * sell_ratio) * stock_price
        stock_price = real[i]
        capital_simulate.append(money + stock_price * share_num)
        capital_stock.append(stock_price * int(10000/real[0]))
    return capital_simulate, capital_stock


