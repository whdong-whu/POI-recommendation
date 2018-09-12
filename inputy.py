# encoding=utf-8

import numpy as np
import tensorflow as tf
import time
from collections import defaultdict
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
def getdata(fl,a,b):
    k=0
    t1 = time.time()
    y={}
    v=np.loadtxt(fl)
    for i in range(a,b):
        for j in range(len(v)):
            y[i,j]=0
            k=k+1
            print (k)
    t2 = time.time()
    print ("cost:", t2 - t1)
    return y

def  readfile(fi):
   user_hash={}
   vocab_hash={}
   with open ( fi,'r') as f:
        for line in f :
           line1=line.strip()
           user=int(line1.split('\t')[0])
           token=int(line1.split('\t')[1])
           user_hash[user]=user_hash.get(user, int(len(user_hash)))
           vocab_hash[token]=vocab_hash.get(token, int(len(vocab_hash)))
   return user_hash,vocab_hash

def readFile(filename):
    t1=time.time()
    f=open(filename,"r")
    contents_lines=f.readlines()
    f.close()
    t2=time.time()
    print ("cost:",t2-t1)
    return contents_lines

def getRatingInformation(ratings):
    t1=time.time()
    rates=[]
    for line in ratings:
        rate=line.split("\t")
        rates.append([float(rate[0]),float(rate[1]),float(rate[2])])
    t2=time.time()
    print ("cost:",t2-t1)
    return rates

def createUserRankDic(rates):
    t1=time.time()
    user_rate_dic={}
    item_to_user={}
    for i in rates:
        user_rank=(i[1],i[2])
        if i[0] in user_rate_dic:
            user_rate_dic[i[0]].append(user_rank)
        else:
            user_rate_dic[i[0]]=[user_rank]
        if i[1] in item_to_user:
            item_to_user[i[1]].append(i[0])
        else:
            item_to_user[i[1]]=[i[0]]
    t2=time.time()
    print ("cost:",t2-t1)
    return user_rate_dic,item_to_user

def  bianhao(fi):
   user_hash={}
   vocab_hash={}
   with open ( fi,'r') as f:
        for line in f :
           line1=line.strip()
           user=int(line1.split('\t')[0])
           token=int(line1.split('\t')[1])
           user_hash[user]=user_hash.get(user, int(len(user_hash)))
           vocab_hash[token]=vocab_hash.get(token, int(len(vocab_hash)))
        return  user_hash,vocab_hash

def func ():
    n=1
    data_dir = 'G:/hecheng/Foursquare/'
    train_file = data_dir + 'Foursquare_train.txt'
    test_contents=readFile(train_file)
    test_rates=getRatingInformation(test_contents)
    test_dic,test_item_to_user=createUserRankDic(test_rates)
    user_hash, vocab_hash=bianhao(train_file)
    fu='G:/hecheng/u.txt'
    fv='G:/hecheng/v.txt'
    u = np.loadtxt(fu)
    for o in range (0,1):
        q='y.bin'
        p=str(n)+q
        a=o*50
        b=a+50
        LY=getdata(fv,a,b)
        for i in test_dic:
            for j in range(len(test_dic[i])):
                uid = user_hash[i]
                lid = int(test_dic[i][j][0])
                lid = vocab_hash[lid]
                LY[uid, lid] = 1
        sorted_LY = sorted(LY.items(), key=lambda k: k[0])
        Y = []
        for j in range(len(sorted_LY)):
            Y.append(sorted_LY[j][1])
        np.savetxt(p, Y)
        Y= np.array(Y, dtype='float32')
        Y.tofile(p)
        n+=1
func()