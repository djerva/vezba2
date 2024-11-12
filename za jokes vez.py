import pandas as pd
jokes_df = pd.read_csv('shortjokes.csv')
# Extract jokes text
jokes = jokes_df['Joke'].head(5000).values
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

#Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(jokes)
total_words = len(tokenizer.word_index)+1
#Create input sequences and labels
input_sequence = []
for joke in jokes:
  token_list = tokenizer.texts_to_sequences([joke])[0]
  for i in range (1,len(token_list)):
    n_gram = token_list[:i+1]
    input_sequence.append(n_gram)
    
max_sequence_length = max([len(i) for i in input_sequence])

#padding of input sequence to max length
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequence = pad_sequences(input_sequence, maxlen = max_sequence_length, padding='pre')

X = padded_input_sequence[:,:-1]
y = padded_input_sequence[:,-1]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y,num_classes=total_words)

#Build LSTM model

from keras import Sequential
from keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(total_words,100,input_length=max_sequence_length-1))
model.add(LSTM(150))
model.add(Dense(total_words,activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # Train the model
model.fit(X, y, epochs=30, verbose=1)

# Generate a new joke based on a seed text
import numpy as np
def generate_joke(seed_text, next_words, model, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Seed text to start the joke
seed_text = "Why did the chicken cross the road"

# Number of words to generate in the joke
next_words = 10

# Generate and print the new joke
generated_joke = generate_joke(seed_text, next_words, model, max_sequence_length)
print("Generated Joke:", generated_joke)