import nltk
import emoji


stop_words = set(nltk.corpus.stopwords.words('english'))
english_words = set(nltk.corpus.words.words())
stop_words.add('food')
lem = nltk.stem.WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')

def text_cleaner(text):

    text = emoji.demojize(text, delimiters=("", "")) # demojize the emojis in the docs

    text = text.lower() # to lowercase
    
    text = tokenizer.tokenize(text) # tokenize with regular expressions

    text = [w for w in text if w not in stop_words] # remove stopwords

    text = [w for w in text if w in english_words] # keep only english words

    text = [lem.lemmatize(w) for w in text] # lemmatize

    text = [w for w in text if len(w) > 3] # keep only words longer than 3 characters

    return text