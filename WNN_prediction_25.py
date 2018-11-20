import re  
import sys  
import math
import numpy as np
import matplotlib.pyplot as plt

def readArff(fileName):  ##只读没有‘？’的数据
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


def cross_vlid_file_split(r,data):  #  交叉验证，c代表第几行
    #训练集合
    # print("数据行数总：",len(data))
    test_data=[]
    train_data=[]

    test_data.append(data[r])#留一法，此行作为测试集，剩下的作为训练集
    for index in range(r):
        train_data.append(data[index])
    for index2 in range(r+1,len(data)):
        train_data.append((data[index2]))
        ##将数据转float
    for i in range(len(train_data)):
        for j in range(len(train_data[0])-1):
            train_data[i][j]=float( train_data[i][j])
    for i in range(len(test_data)):
        for j in range(len(test_data[0])-1):
            test_data[i][j]=float( test_data[i][j])
    ##输出训练集，测试集，已经测试集原来的label
    # print_data(test_data)
    return train_data, test_data,test_data[0][-1] #lab 用来测试正确率
    #
  ##计算词频，c：列数，这里仅计算非数字的列数, 如:make,
def attribute_frequency(data,c):
    dic={}
    temp_List=[]
    # print(data[:,i])
    for i in range(len(data)):
        temp_List.append(data[i][c])
    for element in temp_List:
        if element not in dic.keys():
            dic.update({element:1})
        else:
            dic[element]+=1
    total_value=sum(dic.values())

    for key in dic:
        dic[key]=dic[key]/total_value
    for index in range(len(data)):
        data[index][c]=dic[data[index][c]]
    # print(data)
    return data

##归一，有取值范围的列数
def attribute_guiyi(data,index2,range_list):
    # new_data=[[]]
    for i in range(len(data)):
        # print(data[i][index2],range_list[0])
        data[i][index2]=((float(data[i][index2])-range_list[0]))/(range_list[-1]-range_list[0])
    # print(new_data)
    return data

def print_data(data):   #用这个打印， 别直接print
    for i in range(len(data)):
        print(data[i])
        


        
def caculate_distance(K,test_data,train_data):#calculate all the fxxking distances
    
    distance=[]
    tmp_dist=0.0
    dist_label=[]
    for i in range(len(train_data)):
        label=train_data[i][-1] # 类别，为价格，在倒数第二个
        # print(len(test_data[0]))
        for j in range(len(test_data[0])-1):
            # print(train_data[i][j]-test_data[0][j])
            chazhi=train_data[i][j]-test_data[0][j]
            tmp_dist+=chazhi*chazhi #(x-x)^2+.... test line zhiyou yihang

        # weighted_dis=gaussian(math.sqrt(tmp_dist),a=1.0,b=0.0,c=0.3)
        dist_label.append(math.sqrt(tmp_dist))##每一个的距离
        dist_label.append(label)
        distance.append(dist_label)
        dist_label=[]
        tmp_dist=0 #updata the tmp list
    #print_data(distance)
    distance.sort()# 升序排列吗找距离最小的K个
    # print(distance)
    return(distance[:K])# 返回前K个 distacne
    
    #print_data(distance)    

def judge_max_label(dist_label,weighted_list):
    # symboling_list=[0,0,0,0,0,0,0]
    weighted_avr=0
    total_price=0
    for i in range(len(dist_label)):
        total_price+=float(dist_label[i][-1])*float(weighted_list[i])
    weighted_price=total_price/sum(weighted_list)
    return weighted_price


def KNN_train(K,train_data,test_data):#weight 还没用，以后用
    # predit_list=[]# store the distacen and the {b,g}
    weighted_list=[]
    K_dist_label=caculate_distance(K,test_data,train_data)
    ##前K个距离，和他的label
    print(K_dist_label)
    for i in range(len(K_dist_label)):
        weight=gaussian(K_dist_label[i][0],a=1.0,b=0.0,c=0.3)
        weighted_list.append(weight)

    # print('加权后的',weighted_list)
    predict_price=judge_max_label(K_dist_label,weighted_list)
    # print("predict:  ",predict_label)

    return predict_price

##准确率
def Accuracy(old_labels,predit_lables):
    ori_price=float(old_labels[0])
    pre_price=float(predit_lables[0])
    return 1-(abs(ori_price-pre_price)/ori_price)

def gaussian(K_dist,a=1.0,b=0.0,c=0.3):
    return a*math.e**(-(K_dist-b)**2/(2*c**2))

#mian function
if __name__ =='__main__':  
    fileName='./autos.arff'
    data=readArff(fileName) # list , n rows  35 colum
    # print_data (data)# feature 数
    data=np.array(data)
    data[:,[-2,-1]]=data[:,[-1,-2]]
    data=data.tolist()
    # print_data(data)
    column_number=[1,2,3,4,5,6,7,13,14,16]##需要处理的列的index
    for index in column_number:
        data=attribute_frequency(data,index)
    # new_data=[]
    column_number_guiyi=[0,8,9,10,11,12,15,17,18,19,20,21,22,23,24]
    range_list = [(65, 256), (86.6, 120.9), (141.1, 208.1), (60.3, 72.3), (47.8, 59.8), (1488, 4066), (61, 326),
                  (2.54, 3.94), (2.07, 4.17), (7, 23), (48, 288), (4150, 6600), (13, 49), (16, 54), (-3,3)]
    for index2 in range(len(column_number_guiyi)):
        data=(attribute_guiyi(data,column_number_guiyi[index2],range_list[index2]))
    # print_data(data)

     # 用于画图，找K
    Acc_list = []
    for K in range(1,50):
        temp_acc_list=[]
        for i in range(len(data)):
            old_labels = []
            predit_lables = []
            train_data, test_data, test_org_lable = cross_vlid_file_split(i, data)
            old_labels.append(test_org_lable)
            ##得到每一次的原来价格
            # K=0
            # weight =1
            # para=50
            predit_lable=KNN_train(K,train_data,test_data)
            predit_lables.append(predit_lable)
            # print('预测价格',predit_lables)
            acc=Accuracy(old_labels,predit_lables)
            # print(acc)
            temp_acc_list.append(acc)
            # print('yuanlaijiage',old_labels)
        Acc_list.append(sum(temp_acc_list)/len(temp_acc_list))
    # print(Acc_list)
   #画图， 大家可以优化一下， 确实有点丑
    # plt.title('M=100')
    K_x=np.linspace(1,50,49)
    Acc_y=Acc_list
    # print(len(K_x),len(Acc_y))
    plt.scatter(K_x, Acc_y,label='accuracy',linewidth=1.0,color='r')
    plt.plot(K_x, Acc_y,label='accuracy',linewidth=0.5,color='b')
    plt.xlim((1,50))

    plt.ylim((min(Acc_y)*0.99,max(Acc_y)*1.01))
    # plt.plot( col = "red", pch = 1, bg = "yellow", xlim =K_x, ylim = Acc_list, lwd = 2, xlab = "WEEK", ylab = "STUDENT")
    plt.title('Predict price accuracy with K_number in range 1 to 50')
    plt.xlabel('K_Number')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()
