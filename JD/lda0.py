#%%
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pycaret
import sqlite3
from kiwipiepy import Kiwi
kiwi=Kiwi()
kiwi.prepare()
# %%

#%%
def getPos(txt='안녕하세요 여러분 여러분 반갑습니다.'):
    res=kiwi.analyze(txt)
    badwords=['썅']
    pos=['NNG','NNP','NNB','NR','NP','VV','VA','VX','VCP','VCN','MM','MA','MAJ']
    tokens=[]
    for tkn in res[0][0]:
        for bad in badwords:
            src=list(tkn)
            if(src[0]!=bad):
                #print(src)
                for p in pos:
                    #print(src[1],p)
                    if(src[1].find(p)>-1):
                        if(len(src[0])>1):
                            tokens.append(src[0])
                            break
    return tokens
#getPos()

# %%
# DB에서 자료 가져오기
sql="select comment from NaverNewsComments"
db='../Database Folder/myResultDB.db'
with sqlite3.connect(db) as conn:
    cur=conn.cursor()
    res=cur.execute(sql).fetchall()
txt=''
txts=[]
for r in res:
    if(r[-1]):
        txts.append(r[-1])
    
# 영어식 한글 표현    
#tks=getPos(txts)
len(txts)
#%%
# CBOW : 카운트기반 BOW

def getCBOW(texts=[],opt='CBOW'):
    copus=[]
    #영어화 문서로 변형
    tarrs=[]
    for t in texts:
        tarr=getPos(t)
        tarrs.append(tarr)
        copus.append(' '.join(tarr))
    #print(copus)
    if opt=='CBOW':
        vec=CountVectorizer()
    else:
        vec=TfidfVectorizer()
    vtr=vec.fit_transform(copus)
    cols=[t for t,n,in sorted(vec.vocabulary_.items())]
    return (cols,vtr.toarray(),tarrs)

# %%
cols,data,tarrs=getCBOW(txts,'TFIDF')
# %%
print(tarrs)
# %%
from gensim import corpora
dic=corpora.Dictionary(tarrs) # 사전만들기
print(len(dic))
# gensim 처리를 위한 수치벡터화
copus=[dic.doc2bow(tarr) for tarr in tarrs]
copus
# LDA 모델작성
lda=gensim.models.ldamodel.LdaModel(copus,
                                    id2word=dic, # 타이틀 출력
                                    passes=20, # 학습숫자
                                    num_topics=3) 
# 주제 분석
topics=lda.print_topics(num_words=5)
topics
# %%
#pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gm

gvis=gm.prepare(lda,copus,dic)
gvis
# %%
pyLDAvis.display(gvis)
# %%
# 잘 분리 되었는가?
# 주제별 엔트로피
# Perplexity(PPL): 텍스트 생성(Text Generation) 언어 모델의 성능 평가지표
# 낮을수록 좋다
# coherence: 의미론적 동질성 단어의 거리가 가까이 응집되어 있는정도

#  한쪽을 고정하고 pass를 변형하면서 perplexity 와 coherence를 봄

ntopic=10
cohs=[]
pers=[]
i=0
xs=[]
for ps in range(2,40):
    lda=gensim.models.ldamodel.LdaModel(copus,
                                    id2word=dic, # 타이틀 출력
                                    passes=ps, # 학습숫자
                                    num_topics=ntopic)
    cm=gensim.models.CoherenceModel(model=lda,corpus=copus,coherence='u_mass') 
    coh=cm.get_coherence()
    per=lda.log_perplexity(copus)

    cohs.append(coh)
    pers.append(per)
    xs.append(i)
    i+=1
# %%
plt.plot(cohs)
plt.show()
# %%
plt.plot(pers)
plt.show()
# %%
# 문장비교 -> 유사도 
# 긍부정, 감성분석, 문맥->주제->챗봇, 조어 추정 엔트로피
# %%
# 도배성 댓글 비율 확인, 내용을 분석->도배성 댓글 예측 분류
# 특정인물에 대한 인식
# 패턴 분석

# 같은 nid가 1분 내에 3개 이상 글이 있는 경우 따로 추출
# 초가 없는 경우도 있음
# date  2023.03.01. 07:37:24
# 만약 같은 nid에 여러 nick이 있다면 nick별로 분류
# nid 셀렉트하고 date 다다음 것과 차이를 보고 1분 이하면 3개 다 따로 append 혹은 따로 저장
# 하는 코드


# 성향 분석모델, 테스트, 알고리즘, 제목들간의 관계
# %%
import sqlite3
from datetime import datetime, timedelta

db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

sql = '''
SELECT nid, nick, comment, date
FROM NaverNewsComments
GROUP BY nid, nick
HAVING COUNT(*) >= 3 AND
       MAX(datetime(date) - datetime(MIN(date))) <= 60;
'''

results = cur.execute(sql).fetchall()
conn.close()

for row in results:
    nid, nick, comment, date = row
    print(f"nid: {nid}, nick: {nick}, comment: {comment}, date: {date}")
# %%
#

#%%

db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

sql = '''
SELECT nid, date, comment
FROM NaverNewsComments

'''
#GROUP BY nid
results = cur.execute(sql).fetchall()
conn.close()

output = []

for i in range(len(results)-2):
    nid1, date1, comment1 = results[i]
    nid2, date2, comment2 = results[i+1]
    if nid1  == nid2:
        if (date2 - date1 <= timedelta(minutes=1)):
            date1 = datetime.strptime(date1, "%Y.%m.%d. %H:%M")
            date2 = datetime.strptime(date2, "%Y.%m.%d. %H:%M")
            output.append((nid1, date1, date2, comment1, comment2))



#for item in output:
#    print(item)

print(output)
# %%
date3
# %%
print(nid2,date1,comment1)
# %%

# %%
import sqlite3
from datetime import datetime, timedelta

# Connect to the database
db = '../Database Folder/myResultDB.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

# Select comments with the same NID
sql = '''
SELECT nid, date, comment, nick
FROM NaverNewsComments
ORDER BY nid, date
'''

# Execute the query
results = cur.execute(sql).fetchall()
conn.close()

output = []
temp = []

# Iterate through the comments
for i in range(len(results)-2):
    nid1, date1, comment1, nick1 = results[i]
    nid2, date2, comment2, nick2 = results[i+1]
    nid3, date3, comment3, nick3 = results[i+2]
    
    # If the NIDs are the same and the comments were posted within 1 minute of each other
    if nid1 == nid2 == nid3:
        if "%S" in date1.strftime("%Y.%m.%d. %H:%M:%S") and "%S" in date2.strftime("%Y.%m.%d. %H:%M:%S"):
            date1 = datetime.strptime(date1.strftime("%Y.%m.%d. %H:%M"), "%Y.%m.%d. %H:%M")
            date2 = datetime.strptime(date2.strftime("%Y.%m.%d. %H:%M"), "%Y.%m.%d. %H:%M")
            date3 = datetime.strptime(date3.strftime("%Y.%m.%d. %H:%M"), "%Y.%m.%d. %H:%M")
            if (date3 - date1 <= timedelta(minutes=1)):
                temp.append((nid1, date1, date2, date3, comment1, comment2, comment3, nick1, nick2, nick3))
        else:
            date1 = datetime.strptime(date1.strftime("%Y.%m.%d. %H:%M"), "%Y.%m.%d. %H:%M")
            date2 = datetime.strptime(date2.strftime("%Y.%m.%d. %H:%M"), "%Y.%m.%d. %H:%M")
            date3 = datetime.strptime(date3.strftime("%Y.%m.%d. %H:%M"), "%Y.%m.%d. %H:%M")
            if (date3 - date1 <= timedelta(minutes=1)):
                temp.append((nid1, date1, date2, date3, comment1, comment2, comment3, nick1, nick2, nick3))

    # If the NIDs are different or the comments were posted more than 1 minute apart
    else:
        if len(temp) >= 3:
            output.extend(temp)
        temp = []
    
if len(temp) >= 3:
    output.extend(temp)

# Print the output
for item in output:
    print(item)
# %%
