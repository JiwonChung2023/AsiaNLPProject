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

# 웹드라이버 가져오기
cdriver='./driver/chromedriver.exe'
driver=webdriver.Chrome(cdriver)
driver.set_window_position(0,50)
driver.set_window_size(800, 1300)
url='https://news.naver.com/main/ranking/popularMemo.naver'
driver.get(url)
time.sleep(2) # 딜레이를 통해 웹이 로딩되는 시간을 기다려줌
# 디비 정의 및 함수 정의하기

dfile='./Database Folder/myResultDB.db'
# "select" 구문
def sqlPrs(sql='',d=[],opt=1):
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
def scrapeAll():
    whereatSel='#_COMMENT_NOTICE > p > em'
    titleSel='#title_area > span'
    dateSel='#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span'
    Xp='//*[@id="cbox_module_wai_u_cbox_content_wrap_tabpanel"]/ul/li[{}]'

    whereat=driver.find_element(by=By.CSS_SELECTOR,value=whereatSel).text
    title=driver.find_element(by=By.CSS_SELECTOR,value=titleSel).text
    date=driver.find_element(by=By.CSS_SELECTOR,value=dateSel).text
    href=driver.current_url.split('?')[0]
    for k in range(1,6):
        li=[]
        src=driver.find_element(by=By.XPATH,value=Xp.format(k)).text
        li=src.split('\n')
        ID=li[0]
        try:
            comment=li[3]
        except:
            comment='삭제된 댓글'
        d=[title,whereat,href,ID,date,comment] # 데이터베이스 입력
        sql='insert into newtNcoms (title,whereat,href,ID,date,comment) values(?,?,?,?,?,?)'
        sqlPrs(sql,d,2)

newsSel='#wrap > div.rankingnews._popularWelBase._persist > div.rankingnews_box_wrap._popularRanking > div > div:nth-child({}) > ul > li:nth-child({}) > div > a'

for j in range(1,13):
    try:
        for i in range(1,6):
            try:
                driver.find_element(by=By.CSS_SELECTOR,value=newsSel.format(j,i)).click()
                time.sleep(2)
                scrapeAll()
                driver.back()
                time.sleep(2)
            except:
                pass
                driver.back()
                time.sleep(2)
    except:
        pass