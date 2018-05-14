import matplotlib.pyplot as plt
import pandas as pd

initial_money = 10000.0

random_data = pd.read_csv('../results/random_guess.csv')
sentiment_data = pd.read_csv('../results/sentiment_only.csv')
rnn_data = pd.read_csv('../results/rnn_only.csv')
avg_data = pd.read_csv('../results/avg_individual.csv')
best_data = pd.read_csv('../results/best_individual.csv')

date = list(random_data['Date'])
random = list(random_data['Simulate'])
real = list(best_data['Real'])
if len(real) < len(date):
    for i in range(len(date)-len(real)):
        real.insert(0,initial_money)
sentiment = list(sentiment_data['Simulate'])
if len(sentiment) < len(date):
    for i in range(len(date)-len(sentiment)):
        sentiment.insert(0,initial_money)
rnn = list(rnn_data['Simulate'])
if len(rnn) < len(date):
    for i in range(len(date)-len(rnn)):
        rnn.insert(0,initial_money)
avg = list(avg_data['Simulate'])
if len(avg) < len(date):
    for i in range(len(date)-len(avg)):
        avg.insert(0,initial_money)
best = list(best_data['Simulate'])
if len(best) < len(date):
    for i in range(len(date)-len(best)):
        best.insert(0,initial_money)

fig, ax = plt.subplots()
plt.plot(date,real,label='Real',c='darkorange')
plt.plot(date,random,label='Random',c='purple')
plt.plot(date,sentiment,label='Sentiment',c='c')
plt.plot(date,rnn,label='RNN',c='b')
plt.plot(date,avg,label='Avg',c='r')
plt.plot(date,best,label='Best',c='g')
plt.xticks(range(0,len(date),5),date[0:len(date):5])
plt.xlabel('Date')
plt.ylabel('Capital')
plt.title('Test Return')
plt.legend()
fig.autofmt_xdate()
plt.show()