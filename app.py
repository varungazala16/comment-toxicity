#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow pandas matplotlib sklearn')


# In[2]:


import os
import pandas as pd
import tensorflow as tf
import numpy as np


# In[3]:


df = pd.read_csv('train.csv')


# In[4]:


df.head()


# In[5]:


from tensorflow.keras.layers import TextVectorization


# In[6]:


X = df['comment_text']
y = df[df.columns[2:]].values


# In[7]:


MAX_FEATURES = 200000 # number of words in the vocab


# In[8]:


vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')


# In[9]:


vectorizer.adapt(X.values)


# In[10]:


vectorized_text = vectorizer(X.values)


# In[11]:


dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)


# In[12]:


batch_x, batch_y = dataset.as_numpy_iterator().next()


# In[13]:


train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


# In[14]:


train_generator = train.as_numpy_iterator()


# In[15]:


train_generator.next()


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding


# In[17]:


model = Sequential()

model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))


# In[18]:


model.compile(loss='BinaryCrossentropy', optimizer='Adam')


# In[19]:


model.summary()


# In[20]:


history = model.fit(train, epochs=1, validation_data=val)


# In[21]:


history.history


# In[22]:


input_text = vectorizer('You freaking suck! I am going to hit you.')


# In[23]:


res = model.predict(np.array([input_text])) 


# In[24]:


res=model.predict(batch_x)


# In[27]:


(res > 0.5).astype(int)


# In[28]:


batch_X, batch_y = test.as_numpy_iterator().next()


# In[29]:


(model.predict(batch_X) > 0.5).astype(int)


# In[30]:


res.shape


# In[31]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


# In[32]:


pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# In[33]:


for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


# In[34]:


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# In[35]:


get_ipython().system('pip install gradio jinja2')


# In[38]:


import tensorflow as tf
import gradio as gr


# In[39]:


model.save('toxicity1.h5')


# In[40]:


model = tf.keras.models.load_model('toxicity1.h5')


# In[41]:


input_str = vectorizer('hey i freakin hate you!')


# In[42]:


res = model.predict(np.expand_dims(input_str,0))


# In[43]:


res


# In[44]:


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text


# In[45]:


interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')


# In[46]:


interface.launch(share=True)


# In[ ]:




