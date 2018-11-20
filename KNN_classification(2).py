import re  
import sys  
import math
import matplotlib.pyplot as plt

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
    #训练集合
    print("数据行数总：",len(data))
    train_data=[]
    for i in range(len(data)-m):
        train_data.append(data[i])
    #print_data(train_data)
    #print(train_data[-1]) #correct
    #都是string，转int
    print("trainnig_行数：",len(train_data))
    for i in range(len(train_data)):
        for j in range(len(train_data[0])-1):# 最后一个是标签， 转不了
            train_data[i][j]=float(train_data[i][j])
    #print_data(train_data) # correct
    
    
    
    #测试集合
    test_origl_lab=[]
    test_data=[]
    for line in data[-m:]:
        test_data.append(line) #hou mian M hang
    #print_data(test_data)
    #都是string，转int
    #print(test_data[0][2])# 看数据是否正确
    print("test_行数：",len(test_data))

    for i in range(len(test_data)):
        #test_data[i][-1]='N'
        for j in range(len(test_data[0])):# 最后一个是标签， 转不了
            
            if j!=len(test_data[0])-1:
                test_data[i][j]=float(test_data[i][j])
            if j==len(test_data[0])-1:
                test_origl_lab.append(test_data[i][j])
                test_data[i][j]='N' # 抹去标签

    return train_data, test_data,test_origl_lab #lab 用来测试正确率
    #print_data(test_data)        
  

    
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
        for j in range(len(line_test)-1):
            #print(train_data[i][j]-line_test[j])
            chazhi=train_data[i][j]-line_test[j]
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

    if num_good>num_bad:
        return 'g'
    else:
        return 'b'
    
       







def KNN_train(K,train_data,test_data,weight):#weight 还没用，以后用
    predit_list=[]# store the distacen and the {b,g}
    for i in range(len(test_data)):
        K_dist_label=caculate_distance(K,test_data[i],train_data)
        #判断些label中的G多还是B多。。 return多的就是我们对这一行测试数据的归类
        predict_label=judge_max_label(K_dist_label)
       # print("predict:  ",predict_label)
        predit_list.append(predict_label)
    return predit_list




def Accuracy(test_org_lable,predit_lable):
    correct=0
    for i in range (len(predit_lable)):
        if test_org_lable[i]==predit_lable[i]: #正确率是
            correct+=1
    return correct/len(predit_lable)
        

#mian function
if __name__ =='__main__':  
    fileName=r'C:\Users\user\Desktop\9417\pj\KNN\KNN\ionosphere.arff'  
    data=readArff(fileName) # list , n rows  35 colum
    #print (len(data[0]))# feature 数
    # 把data分成两部分， 我先用前面的所有数据训练， 最后一行数据做测试
    #m 表示后m行作为测试数据
    m=100
    train_data,test_data,test_org_lable=cross_vlid_file_split(m,data)
    K=0
    weight=10
    Acc_list=[]# 用于画图
    #print("K=",K,"M=",m)
    #print("test_org_label:")
    #for i in range(len(test_org_lable)):
      #  print(test_org_lable[i])
    for i in range(100):
        predit_lable=KNN_train(i,train_data,test_data,weight)
        acc=Accuracy(test_org_lable,predit_lable)
        Acc_list.append(acc*100)
        #print("accuracy: ",acc*100,"%")
   #画图， 大家可以优化一下， 确实有点丑 
    plt.title('M=100')
    K_x=range(100)
    Acc_y=Acc_list
    plt.bar(K_x, Acc_y,label="line",linewidth=3,color='r')
    plt.xlabel('K_Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
