from keras.layers import Input, Dense, LSTM, merge,LeakyReLU, GRU, SimpleRNN, Dropout
from keras.models import Sequential, load_model
from keras.utils import plot_model

# Build the model
def generate_model(seq_len, layer_type,layer_num,rnn_unit,dense_unit):
    model = Sequential()
    if layer_num == 1:
        if layer_type == 'LSTM':
            model.add(LSTM(units=rnn_unit, dropout=0.2,input_shape=(seq_len,1)))
        elif layer_type == 'GRU':
            model.add(GRU(units=rnn_unit, dropout=0.2,input_shape=(seq_len,1)))
        elif layer_type == 'SimpleRNN':
            model.add(SimpleRNN(units=rnn_unit, dropout=0.2,input_shape=(seq_len,1)))
        model.add(Dense(units=1))
    else:
        for i in range(layer_num):
            rnn_units = max((rnn_unit / pow(2,i)), 16)
            if i == 0:
                if layer_type == 'LSTM':
                    model.add(LSTM(units=rnn_units, dropout=0.2,input_shape=(seq_len,1),return_sequences=True))
                elif layer_type == 'GRU':
                    model.add(GRU(units=rnn_units, dropout=0.2,input_shape=(seq_len,1),return_sequences=True))
                elif layer_type == 'SimpleRNN':
                    model.add(SimpleRNN(units=rnn_units, dropout=0.2,input_shape=(seq_len,1),return_sequences=True))
            elif i == layer_num - 1:
                if layer_type == 'LSTM':
                    model.add(LSTM(units=rnn_units, dropout=0.2))
                elif layer_type == 'GRU':
                    model.add(GRU(units=rnn_units, dropout=0.2))
                elif layer_type == 'SimpleRNN':
                    model.add(SimpleRNN(units=rnn_units, dropout=0.2))
            else:
                if layer_type == 'LSTM':
                    model.add(LSTM(units=rnn_units, dropout=0.2, return_sequences=True))
                elif layer_type == 'GRU':
                    model.add(GRU(units=rnn_units, dropout=0.2, return_sequences=True))
                elif layer_type == 'SimpleRNN':
                    model.add(SimpleRNN(units=rnn_units, dropout=0.2, return_sequences=True))
            if i == layer_num - 1:
                dense_units = 1
            else:
                dense_units = max((dense_unit / pow(2,i)), 32)
            model.add(Dense(units=dense_units))
    # model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mse')
    return model
