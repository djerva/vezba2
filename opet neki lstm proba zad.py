import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Create a tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on your text data
tokenizer.fit_on_texts([faqs])

#Shows mapping of words with numbers
print(tokenizer.word_index)
print(len(tokenizer.word_index))

     