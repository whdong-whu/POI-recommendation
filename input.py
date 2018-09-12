# encoding=utf-8

import numpy as np
import time
from collections import defaultdict
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def getdata(fi,fl,a,b):
    t1 = time.time()
    x=defaultdict(list)
    u=np.loadtxt(fi)
    v=np.loadtxt(fl)
    for i in range (a,b):#取u全部行与v每一行组合
        for j in range(len(v)):
            x[i, j].extend(u[i])
            x[i, j].extend(v[j])
        t2 = time.time()
        print ("cost:", t2 - t1)
        return x

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
    f=open(filename,"r")
    contents_lines=f.readlines()
    f.close()
    return contents_lines

def getRatingInformation(ratings):
    rates=[]
    for line in ratings:
        rate=line.split("\t")
        rates.append([float(rate[0]),float(rate[1]),float(rate[2])])
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

def func ():
    n=1
    fu = '/home/machine/workspace/mrd/u.txt'
    fv = '/home/machine/workspace/mrd/v.txt'
    u = np.loadtxt(fu)
    print(len(u))

    for o in range (0,(len(u)//1000)+1):
        q='x.txt'
        p=str(n)+q
        a=o*1000
        b=a+1000
        LX=getdata(fu,fv,a,b)
        sorted_LX= sorted(LX.items(), key=lambda k: k[0])
        X=[]
        for i in range (len(sorted_LX)):
            X.append(sorted_LX[i][1])
        np.savetxt(p, X)
        n+=1
func()
#



