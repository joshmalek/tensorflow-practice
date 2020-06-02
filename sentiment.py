import json

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

print(sentences[0])