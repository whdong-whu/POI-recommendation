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

def  creatcomatrix(n,item_to_user,vhash):
    t1=time.time()
    V= np.zeros((n, n))  # 构造矩阵
    for row in range (0,n):
        for col in range (0,n):

           if (row==col):
              if row in item_to_user.keys() :
                V[row,col]=len(item_to_user[vhash[row]])
           else :
                if (row in item_to_user.keys() and col in item_to_user.keys()):
                     inter= list(set( item_to_user[vhash[row]]).intersection(set( item_to_user[vhash[row]])))
                     V[row,col]=len(inter)

                else :
                     V[row,col]=0
    # print V
    t2=time.time()
    print ("cost:",t2-t1)
    return V


def matrix_factorization(V,d=100,steps=5000,alpha=0.001,beta=0.02):
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


def main() :
    data_dir = '/data/disk10/rect/yelptract/'
    size_file = data_dir + 'Yelp_size.txt'
    train_file = data_dir + 'Yelp_train.txt'
    test_file = data_dir + 'Yelp_test.txt'
    poi_file = data_dir + 'Yelp_poi_coos.txt'
    cat_file = data_dir + 'Yelp_poi_categories.txt'
    # filename1=u'E:\\论文\\poi链接\\数据\\Yelp\\Yelp_train2.txt'
    # size_file=u'E:\\论文\\poi链接\\数据\\Yelp\\Yelp_data_size.txt'
    user_num, poi_num, cat_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num, cat_num = int(user_num), int(poi_num), int(cat_num)
    user_hash, vocab_hash=bianhao(train_file)
    hash_vocab={v:k for k,v in vocab_hash.items()}
    test_contents = readFile(train_file)
    test_rates = getRatingInformation(test_contents)
    test_dic, test_item_to_user = createUserRankDic(test_rates)
    n = poi_num
    V1 = creatcomatrix(n, test_item_to_user,hash_vocab)
    model=NMF(n_components=100,alpha=0.001)
    nP1=model.fit_transform(V1)
    nPQ=model.components_
    # nP1, nQ1, result1 = matrix_factorization(V1)
    np.savetxt('v1.txt', nP1)


main()