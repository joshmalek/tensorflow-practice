from tensorflow.keras.preprocessing.text import Tokenizer

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

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)

print()
print() 
print(sequences)
print(test_seq)
print()