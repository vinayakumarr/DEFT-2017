import numpy as np
import pandas as pd

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
np.random.seed(0)
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

if __name__ == "__main__":

    #load data
    train_df = pd.read_csv('data/task1-train.csv', sep='\t', header=0)
    test_df = pd.read_csv('data/task1-test.csv', sep='\t', header=0)

    raw_docs_train = train_df['Phrase'].values
    raw_docs_test = test_df['Phrase'].values
    #raw_docs_train = raw_docs_train.decode("utf8")
    #raw_docs_test = raw_docs_test.decode("utf8")
    #print(raw_docs_test)
    sentiment_train = train_df['Sentiment'].values
    num_labels = len(np.unique(sentiment_train))
    sentiment_test = test_df['Sentiment'].values
    #text pre-processing
    stop_words = set(stopwords.words('french'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('french')

    print "pre-processing train docs..."
    processed_docs_train = []
    #print(raw_docs_train)
    #np.savetxt("traindata.txt",raw_docs_train,fmt="%s")
    for doc in raw_docs_train:
       doc = doc.decode("utf8")
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_train.append(stemmed)
   
    print "pre-processing test docs..."
    processed_docs_test = []
    for doc in raw_docs_test:
       doc = doc.decode("utf8")
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_test.append(stemmed)

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

    dictionary = corpora.Dictionary(processed_docs_all)
    dictionary_size = len(dictionary.keys())
    print "dictionary size: ", dictionary_size 
    #dictionary.save('dictionary.dict')
    #corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print "converting to token ids..."
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))
 
    seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)

    #pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    print(num_labels)
    #y_train_enc = np_utils.to_categorical(sentiment_train,)
    #le = preprocessing.LabelEncoder()
    #le.fit(sentiment_train)
    #v = le.transform(sentiment_train)   
    
    y_train_enc = np_utils.to_categorical(sentiment_train)

    #le1 = preprocessing.LabelEncoder()
    #le1.fit(sentiment_test)
    #v1 = le1.transform(sentiment_test)
    y_test_enc = np_utils.to_categorical(sentiment_test)
    

    #LSTM
    print "fitting LSTM ..."
    model = Sequential()
    model.add(Embedding(dictionary_size, 256, dropout=0.2))
    model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath="logs/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('logs/training_set_iranalysis1.csv',separator=',', append=False)

    model.fit(word_id_train, y_train_enc, nb_epoch=1000, batch_size=64, validation_split=(word_id_test, y_test_enc), verbose=1, callbacks=[checkpointer,csv_logger])
    model.save("logs/lstm_model.hdf5")
    test_pred = model.predict_classes(word_id_test)
    
    #make a submission
    #test_df['Sentiment'] = test_pred.reshape(-1,1) 
    #header = ['PhraseId', 'Sentiment']
    #test_df.to_csv('./lstm_sentiment.csv', columns=header, index=False, header=True)
    accuracy = accuracy_score(v1, test_pred)
    print(accuracy)
    
