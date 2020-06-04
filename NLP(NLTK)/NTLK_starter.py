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

stemmer = PorterStemmer()
stemmed_sent = []

for word in words:
    stemmed_sent.append(stemmer.stem(word))

stemmed_sent = ' '.join(stemmed_sent)
print(stemmed_sent)

#LEMMATIZATION - kind of like stemming?
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_sent = []

for word in words:
    lemmatized_sent.append(lemmatizer.lemmatize(word))

lemmatized_sent = ' '.join(lemmatized_sent)
print(lemmatized_sent)