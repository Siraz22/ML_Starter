import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

example_text = "Hi Tam, how are you doing today? Isn't it a bright sunny day? I see you have been studying for your final exams. Good luck for those."
words = word_tokenize(example_text)
stop_words = set(stopwords.words('english'))

#TOKENIZE WITH STOPWORDS

filtered_sent = []

for word in words:
    if(word not in stop_words):
        filtered_sent.append(word)

print(filtered_sent)

#STEMMING WORDS
from nltk.stem import PorterStemmer
sentences = sent_tokenize(example_text)

stemmer = PorterStemmer()
stemmed_sent = []

for word in words:
    stemmed_sent.append(stemmer.stem(word))

print(stemmed_sent)