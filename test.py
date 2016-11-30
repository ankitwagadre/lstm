# Load LSTM network and generate text
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import random
import sys

def sample_index(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

trsh = sample_index(1,0.2)
# load ascii text and covert to lowercase
filename = "Human-Action_3.txt"
raw_text = open(filename, 'r', errors='ignore').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 25
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model

print("Building the LSTM Network")
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),init='glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
filename = "weights-improvement-06-1.9294.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

output_file = open("output.txt", "w")
# pick a random seed
for jj in range(20):
	start = np.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	#output_file.write("Seed:")
	seed_string = "\"" + ''.join([int_to_char[value] for value in pattern]) + "\""
	output_file.write(seed_string)
	# generate characters
	for i in range(100):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)[0]
		index = np.argmax(prediction)
		next_index = sample_index(index, 0.2)
		result = int_to_char[next_index]
		seq_in = [int_to_char[value] for value in pattern]
		output_file.write(result)
		pattern.append(next_index)
		pattern = pattern[1:len(pattern)]
	output_file.write("\nDone.\n")
