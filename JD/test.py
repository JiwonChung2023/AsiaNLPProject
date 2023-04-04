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
import numpy as np
from numpy import dot
from numpy.linalg import norm

# %%
test=['한미일, 對잠수함·수색구조훈련 돌입... “北수중 핵공격 대응”',
      '문재인에 이재명까지…尹 불참 속 4·3 제주 희생자 추념식에 野 총결집',
      '尹 "4·3 희생자 명예 회복 최선…고통·아픔 보듬어 나갈 것"',
      '경찰 "윤 대통령 처가 공흥지구 특혜 의혹 수사 이달 마무리"',
      '불가능한 내집…서울서 중위소득 구매가능 아파트 100채중 3채' ,
      '문재인은 오후, 이재명은 오전... 제주 4·3행사에 야권 총출동',
      '[단독] “산불현장은 위험하다...여자 공무원들은 집에 가라”',
      '尹지지율, 0.7%p 오른 36.7%…4주 만에 반등[리얼미터]'
      ]
# %%
#%%
# CBOW : 카운트기반 BOW
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
#%%
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
cols,data,tarrs=getCBOW(test,'TFIDF')
tedf=pd.DataFrame(test)
tedf

# %%
# 코싸인함수
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))
# %%
print(cols,data,tarrs)
# %%



# %%
from gensim import corpora
dic=corpora.Dictionary(tarrs) # 사전만들기
print(len(dic))
# gensim 처리를 위한 수치벡터화
copus=[dic.doc2bow(tarr) for tarr in tarrs]
copus
#%%
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
from sklearn.metrics.pairwise import cosine_similarity
#cos_sim(copus,copus)
# %%
copus
# %%

tfidf_matrix = copus.fit_transform()
# %%
cosine_sim = cosine_similarity(copus, copus)






# %%
tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(test)

print(feature_vect_simple.shape)
print(type(feature_vect_simple))

# %%
# TFidfVectorizer로 transform()한 결과는 Sparse Matrix이므로 Dense Matrix로 변환. 
feature_vect_dense = feature_vect_simple.todense()

#첫번째 문장과 두번째 문장의 feature vector  추출
vect1 = np.array(feature_vect_dense[0]).reshape(-1,)
vect2 = np.array(feature_vect_dense[1]).reshape(-1,)

#첫번째 문장과 두번째 문장의 feature vector로 두개 문장의 Cosine 유사도 추출
similarity_simple = cos_similarity(vect1, vect2)
print('문장 1, 문장 2 Cosine 유사도: {0:.3f}'.format(similarity_simple))



# %%




# %%




# %%
import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained Word2Vec model
model = gensim.models.Word2Vec

# Preprocess and tokenize each title
titles = [title.lower().split() for title in test]

# Get the vector for each title
title_vectors = []
for title in titles:
    vectors = []
    for word in title:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            # Ignore words that are not in the vocabulary
            pass
    title_vectors.append(np.mean(vectors, axis=0))

# Compute pairwise cosine similarity between title vectors
cos_sim = cosine_similarity(title_vectors)

# Find the indices of the most similar titles
np.fill_diagonal(cos_sim, -1)  # Set diagonal elements to -1 to exclude self-similarity
most_similar = np.unravel_index(cos_sim.argmax(), cos_sim.shape)

# Print the most similar titles
print(test[most_similar[0]])
print(test[most_similar[1]])
# %%
lda.print_topics(num_words=5)
# %%
cos_sim = cosine_similarity(copus)
# %%
