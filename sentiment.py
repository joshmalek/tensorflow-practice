import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#these are the same thing
data = [json.loads(line) for line in open("sarcasm_headlines_dataset.json", 'r')]

#data = []
#with open("sarcasm_headlines_dataset.json",'r') as f:
#    for line in f:
#        data.append(json.loads(line))



sentences = []
labels = []
urls = []
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# in order to keep our training data seperate from our testing data, we split the data
training_size = 20000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


tokenizer = Tokenizer(num_words=vocab_size,oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,padding=padding_type,truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,padding=padding_type,truncating=trunc_type)

padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)