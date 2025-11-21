from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
words=["jumping","jumps","jumped","running","runner"]
stemmed_words=[stemmer.stem(word)for word in words]
print(stemmed_words)

from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
words=["jumping","jumps","jumped","running","runner"]
lemmatized_words=(lemma.lemmatize(word,pos='v')for word in words)
print(lemmatized_words)