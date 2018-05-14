from keras.utils import plot_model
from keras.models import load_model
model=load_model('../results/best_rnn_model')
plot_model(model, show_shapes=True,to_file='../results/best_rnn_model.png')