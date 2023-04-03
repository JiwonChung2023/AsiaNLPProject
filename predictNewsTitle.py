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
