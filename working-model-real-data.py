
# coding: utf-8

# In[1]:


import librosa

from math import floor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.externals import joblib

import glob

import tensorflow as tf
from multiprocessing import Pool
import os


# In[2]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[3]:


def extractFeatures(audioFile):
    try:
        X, sample_rate = librosa.load(audioFile, duration=5.0)
        
        stft = np.abs(librosa.stft(X))
        mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40) #40
        mfccdelta = librosa.feature.delta(mfcc) #40
        mfccdelta2 = librosa.feature.delta(mfcc, order=2) #40
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate) #12
        chroma_cqt = librosa.feature.chroma_cqt(y=X, sr=sample_rate)#12
        chroma_cens = librosa.feature.chroma_cens(y=X, sr=sample_rate) #12
        rmse = librosa.feature.rmse(y=X) #1
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate) #7
        spectral_centroid = librosa.feature.spectral_centroid(y=X, sr=sample_rate) #1
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate) #1
        spectral_flatness = librosa.feature.spectral_flatness(y=X) #1
        spectral_rolloff =  librosa.feature.spectral_rolloff(y=X, sr=sample_rate) #1
        zcr = librosa.feature.zero_crossing_rate(y=X) #1
        
        #mel = librosa.feature.melspectrogram(X, sr=sample_rate) #128
        #tempogram = librosa.feature.tempogram(y=X, sr=sample_rate) #384
        #average_tempogram = np.mean(tempogram, axis=0).reshape(1, -1)
        
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate) #6
        pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate) 
        pitch = np.mean(pitches, axis=0).reshape(1, -1)
        magnitude = np.mean(magnitudes, axis=0).reshape(1, -1)
        
        pitch_delta = np.mean(librosa.feature.delta(pitches), axis=0).reshape(1, -1)
        pitch_delta2 = np.mean(librosa.feature.delta(pitches, order=2), axis=0).reshape(1, -1)
        magnitudes_delta = np.mean(librosa.feature.delta(magnitudes), axis=0).reshape(1, -1)
        magnitudes_delta2 = np.mean(librosa.feature.delta(magnitudes, order=2), axis=0).reshape(1, -1)
        
        freqs = librosa.core.fft_frequencies(sample_rate)
        salience = librosa.salience(stft, freqs, [1,2,3,4], fill_value=0)
        salience = np.mean(salience, axis=0).reshape(1, -1)
        
        allFeatures = np.concatenate([mfcc, mfccdelta, mfccdelta2, 
                                      rmse,
                                      zcr, 
                                      pitch, magnitude,
                                      salience,
                                      pitch_delta, pitch_delta2, 
                                      magnitudes_delta, magnitudes_delta2
                                     ])
        np.save(audioFile + ".features.npy", allFeatures.T)
    except Exception as err:
        print ("Could not extract for", audioFile, str(err))


# In[4]:


# allWavFiles = glob.glob("/home/administrator/code/audio-ml/working/categorized/**/*.wav", recursive=True)
# print ("Total files: ", len(allWavFiles))
# p = Pool(4)
# p.map(extractFeatures, allWavFiles)
# p.close()
# p.join()


# In[5]:


allFeatureFiles = glob.glob("/home/administrator/code/audio-ml/working/categorized/**/*.features.npy", recursive=True)
allFeatureFiles = shuffle(allFeatureFiles)
lenTrainingData = int(len(allFeatureFiles)*0.90)
allFeatureFiles_train = allFeatureFiles[:lenTrainingData]
allFeatureFiles_test = allFeatureFiles[lenTrainingData:]
print (len(allFeatureFiles), "files", len(allFeatureFiles_train), "train", len(allFeatureFiles_test), "test")


# In[6]:


idx = np.r_[
    0:40,  # mfcc
    40:80, # delta
    80:120, # del delta
    121:122, # zcr
    122:123, # pitch
    123:124, # magnitudes
    124:125, # salience
    125:129 #more
]
n_features = len(idx)
print ("n_features", n_features)
def select_features(file):
    d = np.load(file)
    return d[:,idx]


# In[7]:


baseScaler = MinMaxScaler(copy=False)
max_seq_length = 0


# In[8]:


allData_train = []
seq_lengths_train = []
tl = 0
for aFeatureFile in allFeatureFiles_train:
    data = select_features(aFeatureFile)
    allData_train.append(data)
    tl = tl+len(data)
    seq_lengths_train.append(tl)
    if (len(data)>max_seq_length):
        max_seq_length = len(data)
allData_train = np.concatenate(allData_train, axis=0)
baseScaler.fit_transform(allData_train)
print ("max_seq_length", max_seq_length)
allData_train = np.split(allData_train, seq_lengths_train)
allData_train = allData_train[:-1]


# In[9]:


allData_test = []
seq_lengths_test = []
tl = 0
for aFeatureFile in allFeatureFiles_test:
    data = select_features(aFeatureFile)
    allData_test.append(data)
    tl = tl+len(data)
    seq_lengths_test.append(tl)
    if (len(data)>max_seq_length):
        max_seq_length = len(data)
allData_test = np.concatenate(allData_test, axis=0)
baseScaler.transform(allData_test)
print ("max_seq_length", max_seq_length)
allData_test = np.split(allData_test, seq_lengths_test)
allData_test = allData_test[:-1]


# In[11]:


for i in range(len(allData_train)):
    aData = allData_train[i]
    thislen = len(aData)
    missing_dims = max_seq_length - thislen
    ones = np.ones(missing_dims*n_features).reshape(missing_dims, n_features)
    allData_train[i] = (np.concatenate([aData, ones], axis=0), thislen)


# In[12]:


for i in range(len(allData_test)):
    aData = allData_test[i]
    thislen = len(aData)
    missing_dims = max_seq_length - thislen
    ones = np.ones(missing_dims*n_features).reshape(missing_dims, n_features)
    allData_test[i] = (np.concatenate([aData, ones], axis=0), thislen)


# In[13]:


joblib.dump(baseScaler, "encoder-scaler") 


# In[14]:


def getBatchData(dt, batch_size, batch_num):
    return dt[batch_num*batch_size:(batch_num+1)*batch_size]


# In[15]:


enc_num_cells = 5
enc_num_units = 1024
dec_num_cells = enc_num_cells
dec_num_units = enc_num_units

learning_rate = 0.000005
lr_decay = 0.95
momentum = 0.5
lambda_l2_reg = 0.001
dropout = 0.8


# In[16]:


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, None, n_features])
seq_length_inp = tf.placeholder(tf.int32, [None])
dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='training_dropout')

seq_length_ones = tf.ones(tf.shape(seq_length_inp), tf.int32)
seq_length_out = tf.add(seq_length_inp, seq_length_ones)

X_prepad = tf.pad(X, ((0,0),(1, 0),(0,0)), mode='CONSTANT', constant_values=0)

with tf.variable_scope('Seq2seq', initializer=tf.variance_scaling_initializer(), reuse = tf.AUTO_REUSE):

    # create encoder cells
    enc_cells = []
    for i in range(enc_num_cells):
        enc_cells.append(tf.nn.rnn_cell.BasicLSTMCell(enc_num_units, activation=tf.nn.tanh))
    enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cells)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob = dropout_keep_prob)

    # create decoder cells
    dec_cells = []
    for i in range(dec_num_cells):
        dec_cells.append(tf.nn.rnn_cell.BasicLSTMCell(dec_num_units, activation=tf.nn.tanh))
    dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cells)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob = dropout_keep_prob)


# In[17]:


with tf.variable_scope('Seq2seq', reuse = tf.AUTO_REUSE): 

    train_enc_out, train_enc_state = tf.nn.dynamic_rnn(
        enc_cell, 
        X[::-1],
        dtype = tf.float32,
        sequence_length = seq_length_inp)
    
    train_training_helper = tf.contrib.seq2seq.TrainingHelper(X_prepad, seq_length_out)

    train_decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell, 
        train_training_helper, 
        train_enc_state)

    train_dec_out, train_dec_state, train_dec_out_seq_length = tf.contrib.seq2seq.dynamic_decode(train_decoder)

    train_dec_out_logits = train_dec_out.rnn_output

    train_output_dense = tf.layers.dense(
        train_dec_out_logits,
        n_features)

    train_output_dense_shape_mid = tf.shape(train_output_dense)[1] - 1
    train_output_dense_padded = tf.pad(train_output_dense[:,:-1,:], ((0,0),(0, max_seq_length-train_output_dense_shape_mid),(0,0)), mode='CONSTANT', constant_values=1)
    
    train_loss_0 = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(train_output_dense_padded - X)))
    
    l2 = lambda_l2_reg * sum(
        tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)
    )
    train_loss = train_loss_0 + l2

    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate,decay = lr_decay,momentum = momentum,epsilon = 1e-8)

    gradients, variables = zip(*optimizer.compute_gradients(train_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))
    
    saver = tf.train.Saver()


# In[19]:


num_epochs = 100
batch_size = 128

this_epoch_loss = 0
this_epoch_test_loss = 0
earlier_epoch_loss = 0
n_bad_epoch = 0
best_loss = 100000

n_batches = len(allData_train)//batch_size
n_batches_test = len(allData_test)//batch_size

with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    for epoch in range(num_epochs):
        if(n_bad_epoch>=5):
            break
        training_loss = 0
        for batch_num in range(n_batches):
            print (epoch+1, round(100*batch_num/n_batches,2), end="\r")
            batch_data = getBatchData(allData_train, batch_size, batch_num)
            batch_data = list(zip(*batch_data))
            X_batch = np.array(batch_data[0])
            X_len = np.array(batch_data[1])
            sess.run(train_op, feed_dict={X: X_batch, seq_length_inp:X_len,dropout_keep_prob: dropout})
            this_train_loss = train_loss.eval(feed_dict={X: X_batch, seq_length_inp:X_len})
            training_loss = training_loss + this_train_loss
            if(batch_num == n_batches-1):
                last_batch_loss = this_train_loss
            
        batch_data = getBatchData(allData_test, len(allData_test), 0)
        batch_data = list(zip(*batch_data))
        X_batch = np.array(batch_data[0])
        X_len = np.array(batch_data[1])
        this_test_loss = train_loss.eval(feed_dict={X: X_batch, seq_length_inp:X_len})

        earlier_epoch_test_loss = this_epoch_test_loss
        this_epoch_loss = training_loss/n_batches
        this_epoch_test_loss = this_test_loss
        if(epoch>0 and this_epoch_test_loss>=earlier_epoch_test_loss):
            n_bad_epoch = n_bad_epoch + 1
        else:
            n_bad_epoch = 0
        if(this_test_loss < best_loss):
            best_loss = this_test_loss
            saver.save(sess, "./librosa_auto_encoder.ckpt")
        print ("Epoch:", epoch+1, "Training Loss:", round(this_epoch_loss,2), round(last_batch_loss,2), "Validation Loss:", round(this_epoch_test_loss,2),round(best_loss,2), "Bad Epoch:", n_bad_epoch)


# In[20]:


with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver.restore(sess, "./librosa_auto_encoder.ckpt")
    firstPrinted = False
    allFeatureFiles = glob.glob("/home/administrator/code/audio-ml/working/categorized/**/*.features.npy", recursive=True)

    for aFeatureFile in allFeatureFiles:
        data = select_features(aFeatureFile)
        this_seq_len = len(data)
        missing_dims = max_seq_length - this_seq_len
        ones = np.ones(missing_dims*n_features).reshape(missing_dims, n_features)
        X_batch = np.concatenate([data, ones], axis=0)
        X_batch = X_batch.reshape(1, X_batch.shape[0], X_batch.shape[1])
        X_len = np.array([this_seq_len])
        enc_out = sess.run(train_enc_state, feed_dict={X: X_batch, seq_length_inp:X_len})
        if (firstPrinted==False):
            firstPrinted = True
            a = enc_out[-1].h.flatten()
            for kk in a:
                print (kk, end = ",")
        (fld, f) = os.path.split(aFeatureFile)
        newPath = os.path.join(fld, f.replace("features","rnno"))
        np.save(newPath, enc_out[-1].h.flatten())


# # DNN Classifier

# In[21]:


import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd


# In[22]:


allLabeledFiles = glob.glob("/home/administrator/code/audio-ml/working/650/**/*.wav", recursive=True)
print (len(allLabeledFiles), " files")


# In[25]:


ys = []
Xs = []
for aLabeledFile in allLabeledFiles:
    try:
        f = aLabeledFile.replace("/home/administrator/code/audio-ml/working/650/", "")
        #f = f.replace(".rnno.npy", "")
        f = f.split("/")
        Xs.append(np.load(aLabeledFile.replace("650","categorized").replace(".wav",".wav.rnno.npy")))
        ys.append(f[0])
    except Exception as err:
        print (aLabeledFile)
print (len(Xs), len(ys))


# In[26]:


Xs = np.array(Xs)
ys = np.array(ys)
print (Xs.shape, ys.shape)


# In[27]:


le = preprocessing.LabelEncoder()
yenc = le.fit_transform(ys)


# In[28]:


print (Xs.shape, yenc.shape)
trainingData = np.column_stack([Xs, yenc])
trainingData = shuffle(trainingData)
trainingData.shape


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(trainingData[:,:enc_num_units], trainingData[:,-1], test_size=0.10, stratify=trainingData[:,-1])


# In[30]:


baseScaler2 = StandardScaler(copy=False)
baseScaler2.fit_transform(X_train)
baseScaler2.transform(X_test)


# In[31]:


joblib.dump(baseScaler2, "encoder-classifier") 


# In[32]:


def get_batch(data, batch_size, batch_num):
    start = batch_num * batch_size
    end = (batch_num + 1) * batch_size
    return data[start:end]


# In[33]:


tf.reset_default_graph()

n_inputs = enc_num_units
n_hidden1 = 750
n_hidden2 = 512
n_hidden3 = 250
n_outputs = 4
dropout_rate = 0.2
lambda_l2_reg = 0.001

training = tf.placeholder_with_default(False, shape=(), name='training')

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

he_init = tf.variance_scaling_initializer()

X_drop = tf.layers.dropout(X, dropout_rate, training=training)

hidden1 = tf.layers.dense(X_drop, n_hidden1, name="hidden1", activation=tf.nn.relu, kernel_initializer=he_init)
hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)

hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", activation=tf.nn.relu, kernel_initializer=he_init)
hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)

hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, name="hidden3", activation=tf.nn.relu, kernel_initializer=he_init)
hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)


logits = tf.layers.dense(hidden3_drop, n_outputs, name="outputs", kernel_initializer=he_init)
y_proba = tf.nn.softmax(logits)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_0 = tf.reduce_mean(xentropy, name="loss")

l2 = lambda_l2_reg * sum(
    tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables()
        if not ("noreg" in tf_var.name or "bias" in tf_var.name)
)

loss = loss_0 + l2

learning_rate = 0.0001

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[34]:


n_epochs = 2000
batch_size = 32
n_batches = X_train.shape[0]//batch_size

this_epoch_accuracy = 0
earlier_epoch_accuracy = 0
highest_accuracy = 0
n_bad_epoch = 0

with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        if(n_bad_epoch>=20):
            break
        train_loss = 0
        for batch_num in range(n_batches):
            X_batch = get_batch(X_train, batch_size, batch_num)
            y_batch = get_batch(y_train, batch_size, batch_num)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training:True})
            this_batch_loss = loss.eval(feed_dict={X: X_batch, y: y_batch})
            train_loss = train_loss + this_batch_loss
        acc_valid = accuracy.eval(feed_dict={X: X_test, y: y_test})
        test_loss = loss.eval(feed_dict={X: X_test, y: y_test})
        earlier_epoch_accuracy = this_epoch_accuracy
        this_epoch_accuracy = acc_valid
        if(this_epoch_accuracy<=earlier_epoch_accuracy):
            n_bad_epoch = n_bad_epoch + 1
        else:
            n_bad_epoch = 0
        if(this_epoch_accuracy>highest_accuracy):
            highest_accuracy = this_epoch_accuracy 
            saver.save(sess, "./dnn-classifier.ckpt")
        if (epoch % 100 == 0 or epoch == n_epochs-1 ):
            print(epoch, "Training Loss:", round(train_loss/n_batches,2), "Validation Loss:", round(test_loss,2), "Test accuracy:", round(acc_valid,4), round(highest_accuracy,4), n_bad_epoch)
    print ("Final Accuracy:", round(100*highest_accuracy,2))


# In[35]:


with tf.Session(config=config) as sess:
    init.run()
    saver.restore(sess, "./dnn-classifier.ckpt")
    Z = logits.eval(feed_dict={X:X_test})
    y_pred = np.argmax(Z, axis=1).astype(int)
    print (X_test.shape, y_test.shape, y_pred.shape)


# In[36]:


list(le.classes_)


# In[37]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])


# In[38]:


joblib.dump(le, "label-encoder") 

