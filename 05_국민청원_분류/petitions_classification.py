#!/usr/bin/env python
# coding: utf-8
## # 국민청원 분류하기 - 주목받을만한글 분류 
- 높은 청원참여인원을 기록한 글들의 특징을 학습하여, 그 글들과 유사성을 계산하여 주목받을만한 글인지 판단 
- 청원참여인원이 1천명 넘을것으로 분류된 청원을 결과로 찾음

## # 과정
- 크롤링 (제목,참여인원,카테고리,청원시작,마감일,청원내용)
- 데이터전처리 (공백,특수문자제거)

- 토크나이징
Konlpy - 형태소분석패키지 , Okt클래스 선정, 제목을 형태소단위로 토크나이징(좋,습니다), 내용을 명사단위로 토크나이징하여 df에저장
df_drop - 분석에필요한 df[final]과 label

- 변수생성
청원참여인원이 1천명이상이면 LABEL에 YES붙임, 아니면 NO붙임

- 단어임베딩 (Word2Vec)  (국민청원 10881건에대한 토큰43937개의 100차원임베딩)
문자를 숫자롭 변환하여 컴퓨터가 이해하도록 처리
토큰에인덱스부여하는방법-단어토큰을 숫자로치환한것뿐이므로 토큰간 의미,유사도 파악이어려움/ One-Hot Encoding -성능안좋음
Word2Vec 
- 단어의 의미,유사도반영하여 벡터로 표현 / 특정토큰 근처의 토큰들을 비슷한 위치의 벡터로 표현한다.
- df[total_token]에서 embedding_model불러옴 / embedding_model.wv.most_similar("음주운전") 으로 유사값 찾음

- 실험설계 (학습,평가)
    
# # 2.1 크롤링  (제목,참여인원,카테고리,청원시작,마감일,청원내용)
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


# # 2.2 데이터 전처리 (공백,특수문자제거)
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
- Konlpy - 형태소분석패키지 , Okt클래스 선정, 제목을 형태소단위로 토크나이징(좋,습니다), 내용을 명사단위로 토크나이징하여 df에저장
- df_drop - 분석에필요한 df[final]과 label 만 가져옴

# In[ ]:
from konlpy.tag import Okt
okt = Okt()
df['title_token'] = df.title.apply(okt.morphs)
df['content_token'] = df.content.apply(okt.nouns)

# [파생변수 생성] - 변수생성
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


# # 2.4 단어 임베딩 -Word2Vec 이용하여 문자를 숫자벡터로 변환
# [단어 임베딩]  (국민청원 10881건에대한 토큰43937개의 100차원임베딩)
# In[ ]:
- 문자를 숫자로 변환하여 컴퓨터가 이해하도록 처리
- 토큰에 인덱스 부여하는 방법-단어토큰을 숫자로치환한것뿐이므로 토큰간 의미,유사도 파악이어려움/ One-Hot Encoding -성능안좋음
- Word2Vec 
- 단어의 의미,유사도반영하여 벡터로 표현 / 특정토큰 근처의 토큰들을 비슷한 위치의 벡터로 표현한다.
- df[total_token]에서 embedding_model불러옴 / embedding_model.wv.most_similar("음주운전") 으로 유사값 찾음

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
# [데이터셋 분할 및 csv저장]
# In[ ]:
from numpy.random import RandomState
rng = RandomState()
tr = df_drop.sample(frac=0.8, random_state=rng)        //데이터를 tran, validation set으로 랜덤하게 분할한다. 전체데이터의 80%를 train set으로 지정, 20%를 validation set으로 지정
val = df_drop.loc[~df_drop.index.isin(tr.index)]
tr.to_csv('data/train.csv', index=False, encoding='utf-8-sig')        //csv로  tran, validation set 저장 
val.to_csv('data/validation.csv', index=False, encoding='utf-8-sig')

# [Field클래스 정의]
# In[ ]:
import torchtext
from torchtext.data import Field
def tokenizer(text):            //토크나이저
    text = re.sub('[\[\]\']', '', str(text))
    text = text.split(', ')
    return text
TEXT = Field(tokenize=tokenizer)    //field클래스는 토크나이징, 단어장생성 등을 지원.
LABEL = Field(sequential = False)    //순서가있는데이터인지 

# [csv에서 데이터 불러오기]
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

# [단어장 및 DataLoader 정의] - 임베딩벡터를 가져와 train_iter, validation_iter를 만듬
# In[ ]:
import torch
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
vectors = Vectors(name="data/petitions_tokens_w2v")
TEXT.build_vocab(train, vectors = vectors, min_freq = 1, max_size = None)    // petition_token_w2v 임베딩벡터를 저장
LABEL.build_vocab(train)                //train 데이터의 단어장(Vocab)을 생성한다.
vocab = TEXT.vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter, validation_iter = BucketIterator.splits(        //train,validationset 을 지정한 배치사이즈만큼 로드하여 배치데이터생성함
    datasets = (train, validation),
    batch_size = 8,
    device = device,
    sort = False
)
print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))



# # 2.6 TextCNN  - 단어장 임베딩, 필터통과시켜 피처맵만듬, 풀링레이어생성, logit값반환
# [TextCNN 모델링]
# In[ ]:
import torch.nn as nn   
import torch.optim as optim 
import torch.nn.functional as F 
class TextCNN(nn.Module):  // ### 
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):   // ### vocab_built부터 train데이터로 생성한 단어장, 임베딩벡터의크기, 피처맵이후 생성되는 채널의수, 필터의크기, output클래스의개수        
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(len(vocab_built), emb_dim)                                           //### 단어장을 임베딩한다
        self.embed.weight.data.copy_(vocab_built.vectors)                                              //Word2Vec으로 학습한 임베딩벡터값을 가져온다.
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])     //nn.Conv2d 함수에 임베딩결과를 전달해 필터생성
        self.relu = nn.ReLU()                
        self.dropout = nn.Dropout(0.4)                                                                 //과적합방지 드롭아웃
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)                                   //클래스에대한 Score생성하기위해 fully connected layer생성     
        
    def forward(self, x):                      //TextCNN모델에 데이터를 입력했을때 Output계산하는 과정
        emb_x = self.embed(x)                  //x로받은 임베딩정보 전달
        emb_x = emb_x.unsqueeze(1)             //emb_x의 첫번째축에 차원을 추가한다. 이유는 2차원 텍스트데이터를 모델에 이미지처럼 입력하려면 원을 추가하여 3차원형태로 변환해야하기떄문
        con_x = [self.relu(conv(emb_x)) for conv in self.convs]                 //### self.convs 에는 filter세개가 리스트형태로 있음. output으로 세가지 필터를 각각통과한 결과인 피처맵 3개가 con_x에 리스트형태로 저장됨.
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]      //맥스풀링을 진행해 리스트형태로 이루어진 풀링레이어를 생성한다.
        
        fc_x = torch.cat(pool_x, dim=1)    //1차원풀링벡터 3개를 concat하여 1개의 fully connected layer생성 
        fc_x = fc_x.squeeze(-1)            //10*3 크기를 30*1형태인 fully connected layer생성
        fc_x = self.dropout(fc_x)         
        logit = self.fc(fc_x)              //30*1크기의 fc_x를 fc함수에 통과시켜 1*2크기인 fully connected layer의 propagation logit값을 연산함
        return logit


# [모델 학습 함수 정의]
# In[ ]:
def train(model, device, train_itr, optimizer):
    model.train()                               
    corrects, train_loss = 0.0,0        
    
    for batch in train_itr:
        text, target = batch.text, batch.label      
        text = torch.transpose(text, 0, 1)                  //연산을위해 텍스트데이터를 역행렬로 변환
        target.data.sub_(1)                                 //target의 값을 1씩줄임    
        text, target = text.to(device), target.to(device)   //모델을 학습시키고자 장비에 할당

        optimizer.zero_grad()                           //gradient초기화
        logit = model(text)                             //텍스트데이터를 textCNN의 input으로 이용해 Output계산
    
        loss = F.cross_entropy(logit, target)           //CNN 모델에서 리턴받은 logit값에 softmax 함수를 통과시켜 yes,no로 분류
        loss.backward()                                 //예측값과 실제레이블데이터 비교하여 Negative Log Loss값 계산
        optimizer.step()                                //Softmax, Negative Log Loss값은 torch.nn.functional모듈의 cross_entropy 로 동시에연산
        
        train_loss += loss.item()    
        result = torch.max(logit,1)[1] 
        corrects += (result.view(target.size()).data == target.data).sum()    //TextCNN 모델의 예측값과 레이블데이터 비교후 맞으면 더함
        
    train_loss /= len(train_itr.dataset)
    accuracy = 100.0 * corrects / len(train_itr.dataset)
    return train_loss, accuracy                    //오차, 정확도 반환


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
model = TextCNN(vocab, 100, 10, [3, 4, 5], 2).to(device)           // TextCNN함수를 모델에 주입시킨다. ###
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_test_acc = -1
for epoch in range(1, 3+1):
    tr_loss, tr_acc = train(model, device, train_iter, optimizer)   // 설정값으로 학습후 오차, 정확도 반환 ###
    print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
    
    val_loss, val_acc = evaluate(model, device, validation_iter)    // 평가후 오차, 정확도 반환 ###
    print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))
        
    if val_acc > best_test_acc:
        best_test_acc = val_acc
        
        print("model saves at {} accuracy".format(best_test_acc))
        torch.save(model.state_dict(), "TextCNN_Best_Validation")
    
    print('-----------------------------------------------------------------------------')

