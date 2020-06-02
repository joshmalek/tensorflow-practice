from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
#OOV replaces words it does not know with "out of vocabulary" token, to preserve length
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
#loads words into index
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
#pages word index to construct sequences for each sentence
sequences = tokenizer.texts_to_sequences(sentences)

#padding makes it so that all sentences are the same length, for training the neural network
#to add padding to the end of the sentences, pad_sequences(sequences,padding='post')
padded = pad_sequences(sequences)

print(word_index)
print()  
print(sequences)
print()
print(padded)


