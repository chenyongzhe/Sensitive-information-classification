#encoding='utf-8'
import os
import codecs
import jieba
from  sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pymysql
import numpy as np
import traceback
from sklearn.model_selection import train_test_split
import re
mydir1=str("C:\\Users\\12763\\Desktop\\PYthon代码\\机器学习\\敏感不良非法文本分类\\text\\n")
mydir2=str("C:\\Users\\12763\\Desktop\\PYthon代码\\机器学习\\敏感不良非法文本分类\\text\\p")

p_unm=0
n_num=0





def jiebaclearText(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr="/ ".join(seg_list)
    f_stop = open("stop.txt")
    try:
        f_stop_text = f_stop.read( )
        f_stop_text=f_stop_text
    finally:
        f_stop.close( )
    f_stop_seg_list=f_stop_text.split('\n')
    for myword in liststr.split('/'):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
            mywordlist.append(myword)
    return ' '.join(mywordlist)



##def extract_text(file_dir):
##  text_all=[]
##  filelist=os.listdir(file_dir)  
##  for filename in filelist:
##     ##print(filename)
##     file=os.path.join(file_dir,filename)
##     fileio=codecs.open(file,'r','gbk')
##     word=fileio.read()
##     fcword=jiebaclearText(word)
##     text_all.append(fcword)
##  return text_all

def extract_n_text():
   db=pymysql.connect(host="localhost",port=3307,user="root",passwd="123456",db="sexeducation", charset='utf8')
   cursor =db.cursor()
   text_all=[]
   try:
      sql="select text from sample limit 1,3000".encode("utf8")
      cursor.execute(sql)
      results = cursor.fetchall()
      count=1
      for row in results:
         #print()
         fcword=jiebaclearText(row[0])
         #print(fcword)
         text_all.append(fcword)
         print("完成"+str(count)+"负向样本")
         count=count+1
      print("负向样本完成")
      return text_all
   except:
       traceback.print_exc()
       ##continue


def extract_p_text():
   db=pymysql.connect(host="localhost",port=3307,user="root",passwd="123456",db="sexeducation", charset='utf8')
   cursor =db.cursor()
   text_all=[]
   try:
      sql="select content from article limit 1,200".encode("utf8")
      cursor.execute(sql)
      results = cursor.fetchall()
      count=1
      for row in results:
         #print()
         fcword=jiebaclearText(re.sub(r'</p>','',re.sub(r'<p.*">', '',row[0])))
         #print(fcword)
         text_all.append(fcword)
         print("完成"+str(count)+"正向样本")
         count=count+1
      print("正向样本完成")
      return text_all
   except:
       traceback.print_exc()
       ##continue
    
text1=extract_n_text()
#print(text1)
text2=extract_p_text();

####print(text1)
###text2=extract_text(mydir2)
####print(text2)
##
text3=text1+text2
lable=[]
for i in range(1,3001):
    lable.append(0)
for i in range(1,201):
    lable.append(1)


def tf_idf(corpus):
     vectorizer=CountVectorizer()##该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
     transformer=TfidfTransformer()##该类会统计每个词语的tf-idf权值
     tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))##第一个fit_transform是计算tf-id
     ##word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
     weight=tfidf.toarray()###将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
     return weight

feature=tf_idf(text3)
print(feature.shape)
x_train,x_test,y_train,y_test=train_test_split(feature,lable,test_size=0.3)

model =GaussianNB(priors=None)
##model=svm.SVC()
##model=BernoulliRBM()
##model = RandomForestClassifier(n_estimators=100,n_jobs=-1)
#model=LogisticRegression()
model.fit(x_train, y_train)

y_pre=model.predict(x_test)



from collections import Counter
count=Counter(y_pre==y_test)
bingo=count[True]/len(y_pre)
print("准确率")
print(bingo)
run=1
while run:

  test_text=input("请输入测试语句,回答exit退出")
  if test_text !="exit" :
     corpus_test=[" ".join(jieba.cut(test_text))]
     vectorizer1=CountVectorizer(decode_error='replace')
     vectorizer1.fit_transform(text3)
     transformer=TfidfTransformer()
     #print(vectorizer1.vocabulary_)
     vectorizer=CountVectorizer(decode_error='replace',vocabulary=vectorizer1.vocabulary_)
     tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus_test))
     wei=tfidf.toarray()
     #print(wei)
     answer=model.predict(wei)
     if answer[0]==0:
         print("色情文本")
     else :
          print("非色情文本")
  else :
     run=0

 

