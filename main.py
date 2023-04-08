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

# 코사인 유사도
def cossim(x,y):
    x=np.array(x)
    y=np.array(y)
    csim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return csim

def getPos(txt='안녕하세요 여러분 매우 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['썅']
    pos=['NNG']
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

# "select" 구문
def sqlPrs(sql='',d=[],opt=1):
    dfile='./Database Folder/myResultDB.db'
    with sqlite3.connect(dfile) as conn:
        cur=conn.cursor()
        # select 문의 경우 실행 후 데이터 반환이 필요하다!
        if opt==1:
            res=cur.execute(sql).fetchall()
        elif opt==2:
            # insert: insert into mvreview (sid,title,whereat,href,ID,date,comment) values(?,?,?,?,?,?,?,?)
            res=cur.execute(sql,d)
        else:
            cur.execute(sql)
            res=0
        conn.commit()
    return(res)

if __name__=='__main__':
    
    kiwi=Kiwi()
    test=sqlPrs('select title from newtNcoms')

    qqList=[]
    for p in test:
        qq=p[0]
        qqList.append(qq)
    test=qqList
    #print(text1)
    tests=[]
    cosses=[]
    sameTitle=''
    print('--------읽어본 기사 제목--------')
    yourNews=input('기사 제목을 입력하세요: ')
    test.insert(0,yourNews)
    print(test[0])
    print('-------------------------------')
    cols,data,tarrs=getCBOW(test,'CBOW')
    tedf=pd.DataFrame(data,columns=cols)



    text1=tedf.iloc[:1,:]
    for i,v in tedf.iloc[1:,:].iterrows():
        #print(test[0])
        if (cossim(text1,v)>=0.001):
            if (test[0]!=test[i]):
                if (sameTitle!=test[i]):
                    tests.append(test[i])
                    cosses.append(cossim(text1,v))
                    myZip=list(zip(tests,cosses))
                    sameTitle=test[i]

    for k in myZip:
        print(k,'\n')