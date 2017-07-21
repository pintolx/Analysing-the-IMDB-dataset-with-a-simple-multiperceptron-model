# Load Larger LSTM network and generate text
# Loading important libraries
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#Loading the dataset  and convert it to lowercase
filename = 'wonderland.txt'
raw_text = open(filename).read()
#raw_text = open(filename, encoding='utf8').read()
raw_text = raw_text.lower()

#Mapping unique characters to ints
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

#Summarise loaded dataset
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# Prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX, dataY = [], []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total patterns: ", n_patterns)

# Reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

#Normalize the dataset
X = X/float(n_vocab)

#One hot encode output
y = np_utils.to_categorical(dataY)

#Define the models
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#Loading the weights
filename = "weights-improvement-49-1.2667-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
#Generate characters
for i in range(1000):
    X = numpy.reshape(pattern, (1, len(pattern), 1))
    X = X/float(n_vocab)
    prediction = model.predict(X, verbose=0)	
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone")