import requests
from multiprocessing import Pool
import re 
from bs4 import BeautifulSoup
import pymysql
import traceback


class Spider :
    companycount=0
    kd={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'}
    count=1
    def getHTML(self,url):
              try:
                 r=requests.get(url,headers=self.kd)
                 r.raise_for_status()
                 r.encoding='utf8'
                 return r.text
              except :
                 return "获取网页异常"
    def clawe_one_page(self,url):
        html=self.getHTML(url)
        soup=BeautifulSoup(html,'html.parser')
        div=soup.find_all('div',attrs={'class':"entry-content clearfix"})
        
        for d in div:
            #print(d.text)
            try:
             sql=("insert into sample (text)"+ " VALUES ('"+d.text+"' )").encode("utf8")
             self.cursor.execute(sql)
             self.db.commit()
             print("第"+str(self.count)+"条文本爬取完成")
             self.count=self.count+1
            except:
                traceback.print_exc()
                continue
    def startclawer(self,startpage):
        self.db=pymysql.connect(host="localhost",port=3307,user="root",passwd="123456",db="sexeducation", charset='utf8')
        self.cursor =self.db.cursor()
        print("数据库连接成功")
        url="http://www.4438xx10.com/xiaoshuo/page/"
        for i in range(startpage,310):
              url_tample="http://www.4438xx10.com/xiaoshuo/page/"+str(i)
              self.clawe_one_page(url_tample)



s=Spider()
s.startclawer(1)
