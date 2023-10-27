#!/usr/bin/env python
# coding: utf-8

# # 2.1 크롤링

# [크롤링]

# In[ ]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup 
import time


result = pd.DataFrame()                                    

for i in range(584274, 595226):
    URL = "http://www1.president.go.kr/petitions/"+str(i)
 
    response = requests.get(URL)    
    html = response.text                                   
    soup = BeautifulSoup(html, 'html.parser')           

    title = soup.find('h3', class_='petitionsView_title')
    count = soup.find('span', class_='counter')           

    for content in soup.select('div.petitionsView_write > div.View_write'):
        content                                         

    a=[]
    for tag in soup.select('ul.petitionsView_info_list > li'): 
        a.append(tag.contents[1])

    if len(a) != 0:
        df1=pd.DataFrame({ 'start' : [a[1]],                
                           'end' : [a[2]],                     
                           'category' :  [a[0]],               
                           'count' : [count.text],             
                           'title': [title.text],              
                           'content': [content.text.strip()[0:13000]]                              
                         })

        result=pd.concat([result, df1])                        
        result.index = np.arange(len(result))             

    if i % 60 == 0:                                        
        print("Sleep 90seconds. Count:" + str(i)           
              +",  Local Time:"+ time.strftime('%Y-%m-%d', time.localtime(time.time()))
              +" "+ time.strftime('%X', time.localtime(time.time()))
              +",  Data Length:"+ str(len(result)))        
        time.sleep(90) 


# [크롤링 데이터 확인]

# In[ ]:


print(result.shape)

df = result
df.head()


# [데이터 엑셀로 저장]

# In[37]:


df.to_csv('data/crawling.csv', index = False, encoding = 'utf-8-sig')


# # 2.2 데이터 전처리

# In[ ]:


df.loc[1]['content']  # 전처리 전


# [전처리]

# In[ ]:


import re

def remove_white_space(text):
    text = re.sub(r'[\t\r\n\f\v]', ' ', str(text))
    return text

def remove_special_char(text):
    text = re.sub('[^ ㄱ-ㅣ가-힣 0-9]+', ' ', str(text))
    return text

df.title = df.title.apply(remove_white_space)
df.title = df.title.apply(remove_special_char)

df.content = df.content.apply(remove_white_space)
df.content = df.content.apply(remove_special_char)


# In[ ]:


df.loc[1]['content']  # 전처리 후


# # 2.3 토크나이징 및 변수 생성

# [토크나이징]

# In[ ]:


from konlpy.tag import Okt

okt = Okt()

df['title_token'] = df.title.apply(okt.morphs)
df['content_token'] = df.content.apply(okt.nouns)


# [파생변수 생성]

# In[ ]:


df['token_final'] = df.title_token + df.content_token

df['count'] = df['count'].replace({',' : ''}, regex = True).apply(lambda x : int(x))

print(df.dtypes)

df['label'] = df['count'].apply(lambda x: 'Yes' if x>=1000 else 'No')


# In[ ]:


df_drop = df[['token_final', 'label']]


# In[ ]:


df_drop.head()


# [데이터 엑셀로 저장]

# In[11]:


df_drop.to_csv('data/df_drop.csv', index = False, encoding = 'utf-8-sig')


# # 2.4 단어 임베딩

# [단어 임베딩]

# In[ ]:


from gensim.models import Word2Vec

embedding_model = Word2Vec(df_drop['token_final'], 
                           sg = 1, # skip-gram
                           size = 100, 
                           window = 2, 
                           min_count = 1, 
                           workers = 4
                           )

print(embedding_model)

model_result = embedding_model.wv.most_similar("음주운전")
print(model_result)


# [임베딩 모델 저장 및 로드]

# In[ ]:


from gensim.models import KeyedVectors

embedding_model.wv.save_word2vec_format('data/petitions_tokens_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format('data/petitions_tokens_w2v') # 모델 로드

model_result = loaded_model.most_similar("음주운전")
print(model_result)


# # 2.5 실험 설계

# [데이터셋 분할 및 저장]

# In[ ]:


from numpy.random import RandomState

rng = RandomState()

tr = df_drop.sample(frac=0.8, random_state=rng)
val = df_drop.loc[~df_drop.index.isin(tr.index)]

tr.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
val.to_csv('data/validation.csv', index=False, encoding='utf-8-sig')


# [Field클래스 정의]

# In[ ]:


import torchtext
from torchtext.data import Field

def tokenizer(text):
    text = re.sub('[\[\]\']', '', str(text))
    text = text.split(', ')
    return text

TEXT = Field(tokenize=tokenizer)
LABEL = Field(sequential = False)


# [데이터 불러오기]

# In[ ]:


from torchtext.data import TabularDataset

train, validation = TabularDataset.splits(
    path = 'data/',
    train = 'train.csv',
    validation = 'validation.csv',
    format = 'csv',
    fields = [('text', TEXT), ('label', LABEL)],
    skip_header = True
)

print("Train:", train[0].text,  train[0].label)
print("Validation:", validation[0].text, validation[0].label)


# [단어장 및 DataLoader 정의]

# In[ ]:


import torch
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator

vectors = Vectors(name="data/petitions_tokens_w2v")

TEXT.build_vocab(train, vectors = vectors, min_freq = 1, max_size = None)
LABEL.build_vocab(train)

vocab = TEXT.vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, validation_iter = BucketIterator.splits(
    datasets = (train, validation),
    batch_size = 8,
    device = device,
    sort = False
)

print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))


# # 2.6 TextCNN

# [TextCNN 모델링]

# In[ ]:


import torch.nn as nn   
import torch.optim as optim 
import torch.nn.functional as F 

class TextCNN(nn.Module): 
    
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
        
        super(TextCNN, self).__init__()
        
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)      
    
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        self.relu = nn.ReLU()                
        self.dropout = nn.Dropout(0.4)         
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)     
        
    def forward(self, x):  
      
        emb_x = self.embed(x)           
        emb_x = emb_x.unsqueeze(1)  

        con_x = [self.relu(conv(emb_x)) for conv in self.convs]       

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]    
        
        fc_x = torch.cat(pool_x, dim=1) 
        fc_x = fc_x.squeeze(-1)       
        fc_x = self.dropout(fc_x)         

        logit = self.fc(fc_x)     
        
        return logit


# [모델 학습 함수 정의]

# In[ ]:


def train(model, device, train_itr, optimizer):
    
    model.train()                               
    corrects, train_loss = 0.0,0        
    
    for batch in train_itr:
        
        text, target = batch.text, batch.label      
        text = torch.transpose(text, 0, 1)          
        target.data.sub_(1)                                 
        text, target = text.to(device), target.to(device)  

        optimizer.zero_grad()                           
        logit = model(text)                         
    
        loss = F.cross_entropy(logit, target)   
        loss.backward()  
        optimizer.step()  
        
        train_loss += loss.item()    
        result = torch.max(logit,1)[1] 
        corrects += (result.view(target.size()).data == target.data).sum()
        
    train_loss /= len(train_itr.dataset)
    accuracy = 100.0 * corrects / len(train_itr.dataset)

    return train_loss, accuracy


# [모델 평가 함수 정의]

# In[ ]:


def evaluate(model, device, itr):
    
    model.eval()
    corrects, test_loss = 0.0, 0

    for batch in itr:
        
        text = batch.text
        target = batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        
        logit = model(text)
        loss = F.cross_entropy(logit, target)

        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    test_loss /= len(itr.dataset) 
    accuracy = 100.0 * corrects / len(itr.dataset)
    
    return test_loss, accuracy


# [모델 학습 및 성능 확인]

# In[ ]:


model = TextCNN(vocab, 100, 10, [3, 4, 5], 2).to(device)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = optim.Adam(model.parameters(), lr=0.001)

best_test_acc = -1

for epoch in range(1, 3+1):
 
    tr_loss, tr_acc = train(model, device, train_iter, optimizer) 
    print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
    
    val_loss, val_acc = evaluate(model, device, validation_iter)
    print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))
        
    if val_acc > best_test_acc:
        best_test_acc = val_acc
        
        print("model saves at {} accuracy".format(best_test_acc))
        torch.save(model.state_dict(), "TextCNN_Best_Validation")
    
    print('-----------------------------------------------------------------------------')

