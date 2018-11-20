import re  
import sys  
import math
import matplotlib.pyplot as plt
import numpy as np
def readArff(fileName):  
    arffFile = open(fileName,'r')  
    data = []
    simbol = '?'
    for line in arffFile.readlines():  
        if not (line.startswith('@')):  
            if not (line.startswith('%')):
                if not (line.startswith('?')):
                    if not simbol in line:
                        if line !='\n':  
                            L=line.strip('\n')  
                            k=L.split(',')  
                            data.append(k)                      
                    #print(k)  
    #print(data)
    return data

def check_wenhao(data):
    count=0
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j]=='?':
                count+=1
    return count


def cross_vlid_file_split(m,data):  #  mei feng zhuang hao  后m行作为测试
    train_data=[]
    test_data=[]
    test_data.append(data[m])
    for index in range(m):
        train_data.append(data[index])
    for index2 in range(m+1,len(data)):
        train_data.append((data[index2]))
        ##将数据转float
    for i in range(len(train_data)):
        for j in range(len(train_data[0])-1):
            train_data[i][j]=float( train_data[i][j])
    for i in range(len(test_data)):
        for j in range(len(test_data[0])-1):
            test_data[i][j]=float( test_data[i][j])
    ##输出训练集，测试集，已经测试集原来的label
    return train_data, test_data,test_data[0][-1] #lab 用来测试正确率

  

    
def print_data(data):   #用这个打印， 别直接print
    for i in range(len(data)):
        print(data[i])
def caculate_distance(K,line_test,train_data):#calculate all the fxxking distances
    
    distance=[]
    tmp_dist=0.0
    dist_label=[]
    #print(len(line_test))
    for i in range(len(train_data)):
        label=train_data[i][-1] # 类别
        # print(label)
        for j in range(len(line_test[0])-1):
            #print(train_data[i][j]-line_test[j])
            chazhi=train_data[i][j]-line_test[0][j]
            tmp_dist+=chazhi*chazhi #(x-x)^2+.... test line zhiyou yihang
        dist_label.append(tmp_dist)
        dist_label.append(label)
        distance.append(dist_label)
        dist_label=[]
        tmp_dist=0 #updata the tmp list
    #print_data(distance)
    distance.sort()# 升序排列吗找距离最小的K个
    #print_data(distance)
    return(distance[:K])# 返回前K个 distacne
    
    #print_data(distance)    

def judge_max_label(dist_label):
    num_good=0
    num_bad=0
    for i in range(len(dist_label)):
        
        if dist_label[i][1]=='g':
            num_good+=1
        if dist_label[i][1]=='b':
            num_bad+=1
    print('number of good:', num_good)
    print('number of bad:',num_bad)
    if num_good>=num_bad:


        return 'g'
    else:
        return 'b'
# def gaussian(K_dist,a=1.0,b=0.0,c=0.3):
#     return a*math.e**(-(K_dist-b)**2/(2*c**2))
def KNN_train(K,train_data,test_data):#weight 还没用，以后用
    predit_list=[]# store the distacen and the {b,g}

    K_dist_label=caculate_distance(K,test_data,train_data)
    print('first K distance and their classification:',K_dist_label)
        #判断些label中的G多还是B多。。 return多的就是我们对这一行测试数据的归类
    # for i in range(len(K_dist_label)):
    #    K_dist_label[i][0]=gaussian(K_dist_label[i][0],a=1.0,b=0.0,para)####进行加权
    # print('加权之后',K_dist_label)
    predict_label=judge_max_label(K_dist_label)
    # print(predit_lable)
       # print("predict:  ",predict_label)
    # predit_list.append(predict_label)
    return predict_label
def Accuracy(test_org_lable,predit_lable):
    correct=0
    for i in range (len(predit_lable)):
        if test_org_lable[i]==predit_lable[i]: #正确率是
            correct+=1
    return correct/len(predit_lable)
#main function
if __name__ =='__main__':  
    fileName='./ionosphere.arff'
    data=readArff(fileName) # list , n rows  35 colum
    #print (len(data[0]))# feature 数
    # 把data分成两部分， 我先用前面的所有数据训练， 最后一行数据做测试
    #m 表示后m行作为测试数据
    # m=100
    # K=0

    Acc_list=[]# 用于画图
    #print("K=",K,"M=",m)
    #print("test_org_label:")
    #for i in range(len(test_org_lable)):
      #  print(test_org_lable[i])
    for K in range(1,50):
        acc_tem_list=[]
        for m in range(len(data)):
            old_labels = []
            predit_lable=[]
            train_data, test_data, test_org_lable = cross_vlid_file_split(m, data)
            old_labels.append(test_org_lable)
            # print('测试',test_data,'训练',train_data)
            # para=50
            predit_lable=KNN_train(K,train_data,test_data)
            # print('old',old_labels,'predict',predit_lable)
            acc=Accuracy(old_labels,predit_lable)
            acc_tem_list.append(acc*100)
        Acc_list.append(sum(acc_tem_list)/len(acc_tem_list))
        # print("accuracy: ",acc*100,"%")
   #画图， 大家可以优化一下， 确实有点丑 
    K_x=np.linspace(1,50,49)
    Acc_y=Acc_list
    # print(len(K_x),len(Acc_y))
    plt.scatter(K_x, Acc_y,label='accuracy',linewidth=1.0,color='r')
    plt.plot(K_x, Acc_y,label='accuracy',linewidth=0.5,color='b')
    # plt.xlim((1,50))

    plt.ylim((min(Acc_y)*0.99,max(Acc_y)*1.01))
    # plt.plot( col = "red", pch = 1, bg = "yellow", xlim =K_x, ylim = Acc_list, lwd = 2, xlab = "WEEK", ylab = "STUDENT")
    plt.title('Classification accuracy with K_number in range 1 to 50')
    plt.xlabel('K_Number')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()
