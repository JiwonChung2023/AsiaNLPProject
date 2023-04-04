#%%
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kiwipiepy import Kiwi
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

kiwi=Kiwi()

def getPos(txt='안녕하세요 여러분 매우 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['썅']
    pos=['NNG','NNP','NNB','NR','NP','VV','VA','VX','VCP','VCN','MM','MA','MAJ']
    tokens=[]
    for tkn in res[0][0]:
        for bad in badwords:
            src=list(tkn)
            if(src[0]!=bad):
               # print(src)
                for p in pos:
                    if(src[1].find(p)>-1):
                        tokens.append(src[0])
    return tokens

def getCBOW(texts=[],opt='CBOW'):
    copus=[]
    # 영어화 문서로 변형
    tarrs=[]
    for t in texts:
        tarr=getPos(t)
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    if opt=='CBOW':    
        vec=CountVectorizer()   
    else:
        vec=TfidfVectorizer()
    vtr=vec.fit_transform(copus)
    cols=[t for t,n,in sorted(vec.vocabulary_.items())]
    return (cols,vtr.toarray(),tarrs)

test=['문재인에 이재명까지…尹 불참 속 4·3 제주 희생자 추념식에 野 총결집',
'尹지지율, 0.7%p 오른 36.7%…4주 만에 반등[리얼미터]',
'[단독] “산불현장은 위험하다...여자 공무원들은 집에 가라”',
 '문재인은 오후, 이재명은 오전... 제주 4·3행사에 야권 총출동',
 '"불가능한 내집"…서울서 중위소득 구매가능 아파트 100채중 3채'
 '경찰 "윤 대통령 처가 "공흥지구 특혜 의혹" 수사 이달 마무리"',
 '尹 "4·3 희생자 명예 회복 최선…고통·아픔 보듬어 나갈 것"',
 '한미일, 對잠수함·수색구조훈련 돌입... “北수중 핵공격 대응”'
 ]

cols,data,tarrs=getCBOW(test,'CBOW')
tedf=pd.DataFrame(data,columns=cols)

# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

text1=tedf.iloc[:1,:]
#print(text1)
print(test[0])
for i,v in tedf.iloc[1:,:].iterrows():
    #print(test[0])
    if (cossim(text1,v)>=0.001):
        print(test[i],':',cossim(text1,v))

# %%
# %%
### 일단 작동은 됨
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kiwipiepy import Kiwi
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

kiwi=Kiwi()

def getPos(txt='안녕하세요 여러분 매우 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['썅']
    pos=['NNG','NNP','NNB','NR','NP','VV','VA','VX','VCP','VCN','MM','MA','MAJ']
    tokens=[]
    for tkn in res[0][0]:
        for bad in badwords:
            src=list(tkn)
            if(src[0]!=bad):
               # print(src)
                for p in pos:
                    if(src[1].find(p)>-1):
                        tokens.append(src[0])
    return tokens

def getCBOW(texts=[],opt='CBOW'):
    copus=[]
    # 영어화 문서로 변형
    tarrs=[]
    for t in texts:
        tarr=getPos(t)
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    if opt=='CBOW':    
        vec=CountVectorizer()   
    else:
        vec=TfidfVectorizer()
    vtr=vec.fit_transform(copus)
    cols=[t for t,n,in sorted(vec.vocabulary_.items())]
    return (cols,vtr.toarray(),tarrs)

db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
titles = df['title'].tolist()


testaaa=input('실행시키면 위에 키워드 입력')
titles.insert(0,testaaa)

test=titles


cols,data,tarrs=getCBOW(test,'CBOW')
tedf=pd.DataFrame(data,columns=cols)


# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

text1=tedf.iloc[:1,:]
#print(text1)
print(test[0])
for i,v in tedf.iloc[1:,:].iterrows():
    #print(test[0])
    if (cossim(text1,v)>=0.001):
        print(test[i],':',cossim(text1,v))




# %%
titles
# %%
####################################################################
from tkinter import *
win = Tk() # 창생성
win.geometry("1000x500")
win.title("test")
win.option_add("*Font","맑은고딕 25")

ent = Entry(win)
ent.pack()

def ent_p():
    a = ent.get()
    btn.config(side='top')
    print(a)

btn = Button(win)
btn.config(text="1번 ent 버튼", width=20,height=5)
btn.config(command=ent_p)
btn.pack(side='left')


def alert():
    aaa="누르면 나오는거"
    btn.config(text=aaa)

btn = Button(win)
btn.config(text="2023-04-04", width=20,height=5)
btn.config(command=alert)

btn.pack(side='right')


win.mainloop() # 창실행
 # %%
def alert():
    print("버튼이 눌림")
# %%
alert("3")
# %%
## 하려다가 포기
def getPos1(title_words, comment_words):
    pos = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MA', 'MAJ']
    tokens = []
    for tkn, tag in zip(title_words + comment_words):
        if tag in pos:
            tokens.append(tkn)
    return tokens


def getCBOW1(texts=[],opt='CBOW'):
    copus=[]
    # 영어화 문서로 변형
    tarrs=[]
    for t in texts:
        title_words, comment_words = t[0].split(), t[1].split()
        tarr=getPos1(title_words, comment_words)  # <-- pass both title_words and comment_words to getPos1()
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    if opt=='CBOW':    
        vec=CountVectorizer()   
    else:
        vec=TfidfVectorizer()
    vtr=vec.fit_transform(copus)
    cols=[t for t,n,in sorted(vec.vocabulary_.items())]
    return (cols,vtr.toarray(),tarrs)



sql = "SELECT DISTINCT title, comment FROM newtNcoms"
db = '../Database Folder/myResultDB.db'
with sqlite3.connect(db) as conn:
    cur = conn.cursor()
    cur.execute(sql)
    res = []
    row = cur.fetchone()
    while row is not None:
        res.append(row)
        row = cur.fetchone()

cols, data, tarrs = getCBOW1(res, 'CBOW')
tedf = pd.DataFrame(data, columns=cols)
#%%
### 새로운 시도
import sqlite3
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load stopwords

stopwords = ['썅']

# Load stemmer
stemmer = nltk.SnowballStemmer('english')

# Connect to the database
db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

# Extract titles from the database
sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)

# Preprocess the titles
def preprocess_text(text):
    # Remove punctuation and digits
    text = re.sub('[^A-Za-z]+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    # Stem words
    words = [stemmer.stem(word) for word in words]
    # Join words
    text = ' '.join(words)
    return text

df['title_processed'] = df['title'].apply(preprocess_text)

# Vectorize the preprocessed titles
vectorizer = TfidfVectorizer()
vectorized_titles = vectorizer.fit_transform(df['title_processed'])

# Calculate cosine similarity between titles
cosine_similarities = cosine_similarity(vectorized_titles)

# Recommend similar titles
title_index = 0  # Example title index
similar_titles = pd.Series(cosine_similarities[title_index]).sort_values(ascending=False)
recommended_titles = similar_titles.iloc[1:6]  # Get top 5 similar titles excluding the same title
recommended_titles = df.iloc[recommended_titles.index]['title'].values
print(recommended_titles)

#%%
#%%
# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

text1=tedf.iloc[:1,:]
#print(text1)
print(test[0])
for i,v in tedf.iloc[1:,:].iterrows():
    #print(test[0])
    if (cossim(text1,v)>=0.001):
        print(test[i],':',cossim(text1,v))

# %%
sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
df
 # %%
cols,data,tarrs=getCBOW(df,'CBOW')
tedf=pd.DataFrame(data,columns=cols)

# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

text1=tedf.iloc[:1,:]
#print(text1)
print(test[0])
for i,v in tedf.iloc[1:,:].iterrows():
    #print(test[0])
    if (cossim(text1,v)>=0.001):
        print(test[i],':',cossim(text1,v))
 # %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# getCBOW 함수 수정
def getCBOW(texts=[], opt='CBOW'):
    copus = []
    tarrs = []
    for t in texts:
        tarr = getPos(t)
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    if opt == 'CBOW':    
        vec = CountVectorizer()   
    else:
        vec = TfidfVectorizer()
    vtr = vec.fit_transform(copus)
    cols = [t for t, n in sorted(vec.vocabulary_.items())]
    return (cols, vtr.toarray(), tarrs, vec)

# recommend_similar_titles 함수 수정
def recommend_similar_titles(title, vectorized_titles, df, threshold=0.001, top_n=10):
    # Get the vector for the input title
    _, input_vector, _, vec = getCBOW([title])
    # Calculate cosine similarity between the input title and all other titles
    cosine_similarities = cosine_similarity(input_vector, vectorized_titles)[0]
    # Sort the similarities in descending order
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    # Get the titles and scores that meet the threshold
    similar_titles = [(df['title'].iloc[i], cosine_similarities[i]) for i in sorted_indices if cosine_similarities[i] >= threshold][:top_n]
    return similar_titles

# Example usage
db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
titles = df['title'].tolist()
cols, data, _, vec = getCBOW(titles)
vectorized_titles = data
input_title = "딸기 재배하는 법"
recommended_titles = recommend_similar_titles(input_title, vectorized_titles, df, threshold=0.2, top_n=5)
for title, score in recommended_titles:
    print(f"{title}: {score}")
# %%
db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
titles = df['title'].tolist()
titles
# %%
cols,data,tarrs=getCBOW(titles,'CBOW')
tedf=pd.DataFrame(data,columns=cols)

# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

text1=tedf.iloc[:1,:]
#print(text1)
print(test[0])
for i,v in tedf.iloc[1:,:].iterrows():
    #print(test[0])
    if (cossim(text1,v)>=0.001):
        print(df['title'][i],':',cossim(text1,v))

# %%
db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
titles = df['title'].tolist()

cols, data, tarrs = getCBOW(titles, 'CBOW')
tedf = pd.DataFrame(data, columns=cols)

# 코사인 유사도
def cossim(x, y):
    x = np.array(x)
    y = np.array(y)
    csim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return csim

text1 = tedf.iloc[:1, :]
print(titles[0])
for i, v in tedf.iloc[1:, :].iterrows():
    if (cossim(text1, v) >= 0.001):
        print(df['title'][i], ':', cossim(text1, v))
# %%
titles
# %%
### 일단 작동은 됨
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kiwipiepy import Kiwi
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

kiwi=Kiwi()

def getPos(txt='안녕하세요 여러분 매우 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['썅']
    pos=['NNG','NNP','NNB','NR','NP','VV','VA','VX','VCP','VCN','MM','MA','MAJ']
    tokens=[]
    for tkn in res[0][0]:
        for bad in badwords:
            src=list(tkn)
            if(src[0]!=bad):
               # print(src)
                for p in pos:
                    if(src[1].find(p)>-1):
                        tokens.append(src[0])
    return tokens

def getCBOW(texts=[],opt='CBOW'):
    copus=[]
    # 영어화 문서로 변형
    tarrs=[]
    for t in texts:
        tarr=getPos(t)
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    if opt=='CBOW':    
        vec=CountVectorizer()   
    else:
        vec=TfidfVectorizer()
    vtr=vec.fit_transform(copus)
    cols=[t for t,n,in sorted(vec.vocabulary_.items())]
    return (cols,vtr.toarray(),tarrs)

db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
titles = df['title'].tolist()
test=titles

cols,data,tarrs=getCBOW(test,'CBOW')
tedf=pd.DataFrame(data,columns=cols)

# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

text1=tedf.iloc[:1,:]
#print(text1)
print(test[0])
for i,v in tedf.iloc[1:,:].iterrows():
    #print(test[0])
    if (cossim(text1,v)>=0.001):
        print(test[i],':',cossim(text1,v))
# %%

# %%
### 일단 돌아는 감
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

kiwi=Kiwi()

def getPos(txt='안녕하세요 여러분 매우 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['썅']
    pos=['NNG','NNP','NNB','NR','NP','VV','VA','VX','VCP','VCN','MM','MA','MAJ']
    tokens=[]
    for tkn in res[0][0]:
        for bad in badwords:
            src=list(tkn)
            if(src[0]!=bad):
               # print(src)
                for p in pos:
                    if(src[1].find(p)>-1):
                        tokens.append(src[0])
    return tokens

def getCBOW(texts=[],opt='CBOW'):
    copus=[]
    # 영어화 문서로 변형
    tarrs=[]
    for t in texts:
        tarr=getPos(t)
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    if opt=='CBOW':    
        vec=CountVectorizer()   
    else:
        vec=TfidfVectorizer()
    vtr=vec.fit_transform(copus)
    cols=[t for t,n,in sorted(vec.vocabulary_.items())]
    return (cols,vtr.toarray(),tarrs)

db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)

sql = "SELECT DISTINCT title FROM newtNcoms"
df = pd.read_sql_query(sql, conn)
titles = df['title'].tolist()

cols, data, tarrs = getCBOW(titles, 'CBOW')
tedf = pd.DataFrame(data, columns=cols)

# 코사인 유사도
def cossim(x, y):
    x = np.array(x)
    y = np.array(y)
    csim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return csim

# Select top 5 similar titles
query_title = titles[1]
similar_titles = []
text1 = tedf.iloc[0, :]
for i, v in tedf.iloc[1:, :].iterrows():
    if cossim(text1, v) >= 0.001:
        similar_titles.append(titles[i])
    if len(similar_titles) == 5:
        break

print(f"Query Title: {query_title}")
print(f"Similar Titles: {similar_titles}")
# %%
