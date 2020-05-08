import numpy as np

def create_list(setlen,setnum,list_num):
    np.random.seed(10)
    training_set=[]
    for i in range(list_num):
        x=[]
        for j in range(setnum-1):
            tmp=list(set(np.random.randint(1,10,size=setlen).tolist()))
            for p in range(len(tmp),setlen):
                tmp.append(0)
        ## 10 means "&" and 11 means "-"
            x=x+tmp+np.random.randint(10,12,size=1).tolist()
        tmp=list(set(np.random.randint(1,10,size=setlen).tolist()))
        for p in range(len(tmp),setlen):
            tmp.append(0)
        x=x+tmp
        training_set.append(x)
    
    #print(training_set)
    training_result=res(training_set,setnum,setlen)
    
    return training_set, training_result
    

def res(x,setnum,setlen):
    y=[]
    for i in range(len(x)):
        xi=x[i]
        ## initialize yi=all the elements in x[0]
        yi=[]
        j=0
        while ((xi[j]>0) & (xi[j]<10)):
            yi.append(xi[j])
            j+=1
        
        for k in range(setnum-1):
            j=(k+1)*(setlen+1)
            tmp=[]
            while ((j<len(x[0])-1) & (xi[j]>0)& (xi[j]<10)):
                tmp.append(xi[j])
                j=j+1
            if ((xi[j]>0) & (xi[j]<10)):
                tmp.append(xi[j])
            
            temp=yi
            tmp.sort()
            temp.sort()
            tempy=[]
            q=0
            '''
            print(xi)
            print("p",temp)
            print(tmp)
            '''
            for m in range(len(temp)):
                while ((temp[m]>tmp[q]) & (q<len(tmp)-1)):
                    q+=1
                    #print(q)
                    if (q==len(tmp)-1):
                        break
                
                if ((xi[(k+1)*(setlen+1)-1]==10) & (temp[m]==tmp[q])):
                    tempy.append(temp[m])
                if ((xi[(k+1)*(setlen+1)-1]==11) & (temp[m]!=tmp[q])):
                    tempy.append(temp[m])
               # print(tempy)
                        
            yi=tempy
            
        for n in range(len(yi),setlen):
            yi.append(0)
        
        y.append(yi)
    return y

'''
x,y=create_list(4,3,3)
print(x)
print(y)
'''