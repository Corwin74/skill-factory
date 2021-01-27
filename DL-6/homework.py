import ktrain
from ktrain import text
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# MODEL
#BATCH_SIZE  = 32
EPOCH       = 4
VAL_SPLIT   = 0.01  #15%

# TOKENIZER
# The maximum number of words to be used. (most frequent)
MAX_WORDS = 20000
# Max number of words in each complaint.
MAXLEN    = 120

DATA_PATH = 'data/'
PATH      = 'working/'
train = pd.read_csv(DATA_PATH+'train.csv',)
test = pd.read_csv(DATA_PATH+'test.csv',)
# Подготовим таргеты
classes =  list(set(train.genre))
#Y = pd.get_dummies(train.genre)
#classes =  Y.columns
train = train[['text', 'genre']]
train = pd.get_dummies(train, prefix='', prefix_sep='', columns=['genre',])
train.to_csv('train_df.csv', index=False)
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(train_df=train, 
                                                                   text_column='text',
                                                                   label_columns=classes,
                                                                   val_pct=VAL_SPLIT, 
                                                                   max_features=MAX_WORDS, 
                                                                   maxlen=MAXLEN,
                                                                   preprocess_mode='bert',
                                                                   ngram_range=1)
model = text.text_classifier('bert', (x_train, y_train), preproc=preproc, )
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))
learner.autofit(2e-5, EPOCH)
learner.save_model('model_1')
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('predictor_1')
predict = predictor.predict(test.text.values)
submission = pd.DataFrame({'id':range(1, len(predict)+1),'genre':predict},columns=['id', 'genre'])
submission.to_csv('submission.csv', index=False)
submission.head()
# на соревнованиях всегда сохраняйте predict_proba, чтоб потом можно было построить ансамбль решений
predict_proba = predictor.predict_proba(test.text.values)
predict_proba = pd.DataFrame(predict_proba, columns=classes)
predict_proba.to_csv('predict_proba.csv', index=False)
