#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycaret
import sqlite3
from kiwipiepy import Kiwi
kiwi=Kiwi()
kiwi.prepare()
# %%
def getPos(txt='안녕하세요 여러분 여러분 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['']
    #    pos=['NNG','NNP','NNB','NR','NP','VV','VA','VX','VCP','VCN','MM','MA','MAJ']
    pos=['NNG']
    tokens=[]
    for tkn in res[0][0]:
        for bad in badwords:
            src=list(tkn)
            if(src[0]!=bad):
                #print(src)
                for p in pos:
                    #print(src[1],p)
                    if(src[1].find(p)>-1):
                        tokens.append(src[0])
                        break
    return tokens
# %%
# DB에서 자료 가져오기
sql="select comment from NaverNewsComments"
db='./Database Folder/myResultDB.db'
with sqlite3.connect(db) as conn:
    cur=conn.cursor()
    res=cur.execute(sql).fetchall()
txt=[]
for r in res:
    txt.append(r[-1])
print(txt)
# 비교할 대상을 설정
txt[0]='조국 조민'
txt
# %%
# 영어식 한글 표현    
tks=getPos(txt)
print(tks)
# %%
# CBOW : 카운트기반 BOW
from sklearn.feature_extraction.text import CountVectorizer
# TF_IDF (Term Frequency - Inverse Document Frequency)
# 숫자기반 문서 벡터 - 단어의 중요성 기반의 벡터
from sklearn.feature_extraction.text import TfidfVectorizer


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

# %%
cols,data,tarrs=getCBOW(txt,'TFIDF')
tedf=pd.DataFrame(data,columns=cols)
print(tedf)
# %%
# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim
#%%
text1=tedf.iloc[:1,:]
print(text1)
corrList=[]
for i,v in tedf.iloc[1:,:].iterrows():
    #print(txt[0])
    #print(txt[i],':',cossim(text1,v))
    if (cossim(text1,v)>=0.1):
        corrList.append(txt[i])
# %%

#%%
text1=tedf.iloc[:1,:]
print(text1)
# %%
for i,v in tedf.iloc[1:,:].iterrows():
    #print(v)
    print(txt[i],':',np.sum(np.square(text1.values-v.values)))
# %%
tedf
# %%
import gensim
#%%
cols,data,tarrs=getCBOW(corrList,'TFIDF')
#%%
print(tarrs)
#%%
from gensim import corpora
dic=corpora.Dictionary(tarrs) # 사전만들기
print(len(dic))

copus=[dic.doc2bow(tarr) for tarr in tarrs]
copus
lda=gensim.models.ldamodel.LdaModel(copus,num_topics=3)
lda.print_topics(num_words=5)
# %%
#pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gm
gvis=gm.prepare(lda,copus,dic)
gvis
# %%
pyLDAvis.display(gvis)
# %%
