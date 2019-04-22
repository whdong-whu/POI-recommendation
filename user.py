#  encoding=utf-8

import numpy as np
import time


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



def  creatcomatrix(n,item_to_user):
    t1=time.time()
    V= np.zeros((n, n))  # 构造矩阵
    for row in range (0,n):
        for col in range (0,n):

           if (row==col):
              if row in item_to_user.keys() :
                V[row,col]=len(item_to_user[row])
           else :
                if (row in item_to_user.keys() and col in item_to_user.keys()):
                     inter= list(set( item_to_user[row]).intersection(set( item_to_user[col])))
                     V[row,col]=len(inter)

                else :
                     V[row,col]=0
    # print V
    t2=time.time()
    print ("cost:",t2-t1)
    return V


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







def main ():
    # file=u'E:\\论文\\poi链接\\数据\\Yelp\\Yelp_train2.txt'
    data_dir = '/home/machine/mrd/Foursquare/'
    size_file = data_dir + 'Foursquare_data_size.txt'
    train_file = data_dir + 'Foursquare_train.txt'
    test_file = data_dir + 'Foursquare_test.txt'
    poi_file = data_dir + 'Foursquare_poi_coos.txt'
    cat_file = data_dir + 'Foursquare_category.txt'
    user_num,poi_num,cat_num=open(size_file,'r').readlines()[0].strip('\n').split()
    user_num,poi_num,cat_num=int(user_num),int(poi_num),int(cat_num)
    user_hash, vocab_hash=bianhao(train_file)
    t1=time.time()
    v1=np.loadtxt("v1.txt")
    v2=np.loadtxt("v2.txt")
    v3=np.loadtxt("v3.txt")
    d=np.hstack((v1,v2,v3))
    t2 = time.time()
    print ("cost:", t2 - t1)
    np.savetxt('v.tx',d)
    test_contents=readFile(train_file)
    test_rates=getRatingInformation(test_contents)
    test_dic,test_item_to_user=createUserRankDic(test_rates)
    U=[]
    t3 = time.time()
    for i in test_dic:
        eu=0
        u=0
        print (i)
        for j in  range (len(test_dic[i])):
             lid=int (test_dic[i][j][0])
             lid=vocab_hash[lid]
             eu=eu+d[lid]
             u=u+1
        eu=eu/u
        U.append(eu)
        # print eu
    t4 = time.time()
    print ("cost:", t4 - t3)
    U=np.array(U)
    print (len(U))
    np.savetxt('u.txt',U)

main ()







