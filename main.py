import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import preprocessor as p # pip install tweet-preprocessor

keras_tokenizer = Tokenizer(num_words=5000)

app = FastAPI()

model_dir = "perf_advanced_tweetprep_glove.h5"
model = load_model(model_dir)

# get input

text = input
text_cleaned = [p.clean(text)]

keras_tokenizer.fit_on_texts(text_cleaned)
token_text = keras_tokenizer.texts_to_sequences(text_cleaned)
pad_text = pad_sequences(token_text, padding='post', maxlen=100)

prediction = model.predict(pad_text)

# return prediction