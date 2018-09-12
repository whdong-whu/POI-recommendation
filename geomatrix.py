#  encoding=utf-8
import numpy as np
from numpy import *
import math
from math import sin,cos,radians, asin, sqrt, acos
import json
import time
from sklearn.decomposition import NMF

def readFile(filename):
    t1=time.time()
    contents_lines=[]
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

def creatgeomatrix  (m,n,lid_dic,hashv):
    t1=time.time()
    V=V= np.zeros((n, n))
    for row in range (0,n):
        for col in range (0,n):

           if (row==col):
              if row in lid_dic.keys() :
                V[row,col]=1
           else :
                if (row in lid_dic.keys() and col in lid_dic.keys()):

                    tlon1=lid_dic[hashv[row]][0][0]
                    tlat1=lid_dic[hashv[row]][0][1]
                    tlon2=lid_dic[hashv[col]][0][0]
                    tlat2=lid_dic[hashv[col]][0][1]
                    if dis(tlon1, tlat1, tlon2, tlat2)<m:

                          V[row,col]=1

                    else :
                          V[row,col]=0
    t2=time.time()
    print ("cost:",t2-t1)
    return V

def dis(lon1, lat1, lon2, lat2):
    lon1,lat1,lon2,lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1

    c = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon)
    r = 6371
    if c > 1:
        c = 1
    return int(r * acos(c))

def matrix_factorization(V,d=100,steps=5000,alpha=0.0002,beta=0.02):
    t1=time.time()
    R=np.array(V)
    N=len(R)
    M=len(R[0])
    P=np.random.rand(N,d) #随机生成一个 N行 K列的矩阵
    Q=np.random.rand(d,M)

    result=[]
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j]) # .dot(P,Q) 表示矩阵内积
                    for k in range(d):
                        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)
                    for k in range(d):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
        result.append(e)
        if e<0.001:
            break
    t2=time.time()
    print ("cost:",t2-t1)
    return P,Q,result

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



def main():
    data_dir = '/data/disk10/rect/yelptract/'
    size_file = data_dir + 'Yelp_size.txt'
    train_file = data_dir + 'Yelp_train.txt'
    test_file = data_dir + 'Yelp_test.txt'
    poi_file = data_dir + 'Yelp_poi_coos.txt'
    cat_file = data_dir + 'Yelp_poi_categories.txt'
    # filename2=u'E:\\论文\\poi链接\\数据\\Yelp\\Yelp_poi_coos 1.txt'
    # size_file=u'E:\\论文\\poi链接\\数据\\Yelp\\Yelp_data_size.txt'
    user_num,poi_num,cat_num=open(size_file,'r').readlines()[0].strip('\n').split()
    user_num,poi_num,cat_num=int(user_num),int(poi_num),int(cat_num)
    user_hash,vocab_hash = bianhao(train_file)
    hash_vocab = {v: k for k, v in vocab_hash.items()}
    test_contents=readFile(poi_file)
    test_rates=getRatingInformation(test_contents)
    lid_dic,test_item_to_user=createUserRankDic(test_rates)
    n=poi_num
    V2=creatgeomatrix(4,n,lid_dic,hash_vocab)
    model = NMF(n_components=100, alpha=0.001)
    nP2 = model.fit_transform(V2)
    nQ2 = model.components_
    # nP2,nQ2,result2=matrix_factorization(V2)
    np.savetxt('v2.txt',nP2)

main()
