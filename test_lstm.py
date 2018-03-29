'''Trains a Minimal RNN on the IMDB sentiment classification task.
The dataset is actually too small for Minimal RNN to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.datasets import imdb
#import gensim, codecs

def build_model():
    model1 = encoder()
    return decoder(model1)

def encoder():
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(Bidirectional(GRU(512, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.2))
    return model

def decoder(model1):
    model = Sequential()
    model.add(Dense(input_shape=(64,), units=512))
    model.add(GRU(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return Model(inputs=[model1], outputs=[model])

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = build_model()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
