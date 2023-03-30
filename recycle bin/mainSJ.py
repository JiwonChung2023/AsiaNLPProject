#%%
# 라이브러리 가져오기

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup as bsp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bsp
import time
#%%
# 웹드라이버 가져오기
cdriver='./driver/chromedriver.exe'
driver=webdriver.Chrome(cdriver)
driver.set_window_position(0,50)
driver.set_window_size(800, 1300)
url='https://media.naver.com/press/001/ranking'
driver.get(url)
time.sleep(2) # 딜레이를 통해 웹이 로딩되는 시간을 기다려줌
# %%
# 디비 정의 및 함수 정의하기

dfile='./db/navernews.db'
# "select" 구문
def sqlPrs(sql='',d=[],opt=1):
    with sqlite3.connect(dfile) as conn:
        cur=conn.cursor()
        # select 문의 경우 실행 후 데이터 반환이 필요하다!
        if opt==1:
            res=cur.execute(sql).fetchall()
        elif opt==2:
            # insert: insert into mvreview (rid,href,point,user,rday,content) values(?,?,?,?,?,?,?)
            res=cur.execute(sql,d)
        else:
            cur.execute(sql)
            res=0
        conn.commit()
    return(res)

def upgradeScrapeComments():
    pInfo=driver.find_element(by=By.CSS_SELECTOR,value='div.u_cbox_userinfo_meta > div.u_cbox_userinfo_meta_header > button')
    pInfoCon=bsp(pInfo.get_attribute('data-param'),'html.parser')
    li1=str(pInfoCon).split(',')
    nick=li1[6].replace(" 'nickName':'",'').split("'")[0] # 닉네임 추출
    nid=li1[7].replace(" 'maskedId':'",'').split("'")[0] # 마스크드 아이디 추출
    comms=driver.find_elements(by=By.CSS_SELECTOR,value='.u_cbox_area')
    jw=0
    for com in comms:
        if (com.text==''):
            pass
        else:
            if (jw==0):
                jw+=1
                pass
            else:
                li2=com.text.split('\n')
                comment=li2[0]
                date=li2[1]
                whereat=li2[3]
                if (date=='댓글모음'):
                    pass
                else:
                    d=[whereat,nid,nick,date,comment] # 데이터베이스 입력
                    sql='insert into projectTable (whereat,nid,nick,date,comment) values(?,?,?,?,?)'
                    sqlPrs(sql,d,2)
        
# 오류 나면 이거 다시 실행시켜라
def clickIt(ssel,drv=driver):
    drv.find_element(by=By.CSS_SELECTOR,value=ssel).click()
    
def goScroll(level=0):
    #dheight=driver.execute_script('return document.documentElement.scrollHeight')
    dheight=driver.execute_script('window.scrollTo(0,{})'.format(level))
# 본 파일
if __name__=='__main__':
    for k in range(1,19): # 각 언론사 홈페이지로 이동하기
        try:
            if (k<10):
                uurl='https://media.naver.com/press/00'+str(k)+'/ranking'
            else:
                uurl='https://media.naver.com/press/0'+str(k)+'/ranking'
            driver.get(uurl)
            time.sleep(2)

            clickIt('#ct > div.press_ranking_home > ul > li:nth-child(2) > a') # 댓글 많은 순 정렬
            time.sleep(2)

            print('\n현재 {}번째 언론사에서 스크레이핑 중'.format(k),end='')

            for i in range(1,21):
                try:
                    print('.',end='') # 몇 번째 기사인지 알려줌

                    driver.find_element(by=By.XPATH,value='//*[@id="ct"]/div[2]/div[2]/ul/li[{}]/a'.format(i)).click() #뉴스 스무 개 차례대로 클릭하기
                    time.sleep(2)
                    
                    clickIt('#cbox_module > div.u_cbox_wrap.u_cbox_ko.u_cbox_type_sort_favorite > div.u_cbox_view_comment > a') # 댓글 더보기 
                    time.sleep(2)
                    try:
                        clickIt('#cbox_module > div.u_cbox_wrap.u_cbox_ko.u_cbox_type_sort_favorite > div.u_cbox_paginate > a') # 댓글 더보기 2
                        time.sleep(2)

                        clickIt('#cbox_module > div.u_cbox_wrap.u_cbox_ko.u_cbox_type_sort_favorite > div.u_cbox_paginate > a') # 댓글 더보기 3
                        time.sleep(2)

                        clickIt('#cbox_module > div.u_cbox_wrap.u_cbox_ko.u_cbox_type_sort_favorite > div.u_cbox_paginate > a') # 댓글 더보기 4
                        time.sleep(2)
                    except:
                        pass
                except:
                    pass

                goScroll() # 맨 위로 스크롤하기
                comButtons=driver.find_elements(by=By.CSS_SELECTOR,value='div.u_cbox_info > span.u_cbox_info_main > button') # 이용자 최근 댓글 창으로 들어가기
                for comB in comButtons:
                    comB.click()
                    time.sleep(3)
                    try:
                        upgradeScrapeComments() # 댓글 가져오기
                    except:
                        pass
                    driver.back()

                driver.back()
                time.sleep(2)
        except:
            pass

    print('*******The program has successfully ended*******')
    #%%

