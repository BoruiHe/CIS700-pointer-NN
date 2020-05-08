import numpy as np
import torch
from torch import optim
from pointer_network import PointerNet
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import  Variable
import create_data as cdata

class Running:
    
    def __init__(self,setlen=6,setnum=3,hidden_size=256,weight_size=128,train_num=40,test_num=80):
        self.input_size=(setlen+1)*setnum-1
        self.embed_size=32
        self.hidden_size=hidden_size
        self.weight_size=weight_size
        self.setlen=setlen
        self.setnum=setnum
        self.num_layers=1
        self.train_num=train_num
        self.test_num=test_num
        self.dict_size=12
        self.net=PointerNet(self.input_size, self.embed_size, self.hidden_size, self.weight_size, \
                            self.setlen, self.num_layers, self.dict_size)
        torch.manual_seed(10)
        if torch.cuda.is_available():
            self.net.cuda()
        
    def importdata(self):
        x,y=cdata.create_list(self.setlen,self.setnum,self.train_num+self.test_num)
        x = Variable(torch.LongTensor(x))
        y = Variable(torch.LongTensor(y))
        self.train_x=x[:self.train_num]
        self.test_x1=x[self.train_num:]
        self.train_y=y[:self.train_num]
        self.test_y1=y[self.train_num:]
        
        test_x2,test_y2=cdata.create_list(self.setlen,self.setnum,3*self.test_num)
        self.test_x2 = Variable(torch.LongTensor(test_x2))[self.test_num:]
        self.test_y2 = Variable(torch.LongTensor(test_y2))[self.test_num:]
        
        print(self.train_x.shape,self.train_y.shape)
        print(self.test_x1.shape,self.test_y1.shape)
        print(self.test_x2.shape, self.test_y2.shape)
    
    def get_trainable_number(self):
         total_num = sum(p.numel() for p in self.net.parameters())
         trainable_num = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
         print('Total:',total_num, 'Trainable:',trainable_num)
     
    def calLoss(self,X,Y):
         # Calculating the loss
         pred = self.net(X)                 # size: [batch,output_size, seq_len]
         #print(pred.shape,Y.shape)
         pred = pred.view(-1, X.size()[1])  # seq_len = 10  size: [batch*output_size,seq_len]
         y = Y.view(-1)                     #size: [batch*output_size]
         return F.nll_loss(pred, y)


    def test(self,X,Y):
        pred = self.net(X)
        y_ = torch.argmax(pred, 2)
        self.evaluate(Y,y_)


    def evaluate(self,y_real,y_pred):
        correct = 0
        summ = y_real.size()[0]
        for i in range(summ):
            if torch.equal(y_real[i],y_pred[i]):
                correct += 1
        print("Accuracy: {0:.2f}%".format(correct/summ*100))


    def learn_curve(self,tr,te1,te2):
        x1 = (np.arange(len(tr))+1).tolist()
        x2 = (np.arange(len(te1))+1).tolist()
        x3 = (np.arange(len(te2))+1).tolist()
        plt.title("The learning curve")
        # training curve
        plt.plot(x1,tr,color ='green',label="training loss")
        # Testing curve (same size)
        plt.plot(x2, te1,color ='red',label="testing loss(same)")
        # Testing curve (different size)
        plt.plot(x3, te2,color ='blue',label="tesing loss(diff)")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("The loss")
        plt.show()


    def train(self,batch=40,n_epochs=100):
        self.net.train()
        optimizer=optim.Adam(self.net.parameters())
        tr_loss=[] # train loss
        te_loss1=[] # test 1 2
        te_loss2=[]
        for epoch in range(n_epochs):
            for i in range(1,self.train_x.size()[0]-batch,batch):
                x=self.train_x[i:i+batch]
                y = self.train_y[i:i+batch]  # size: [batch, output_size]
                loss = self.calLoss(x,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if epoch % 5 == 0:
                print("Epoch  {} : Loss = {}".format(epoch,loss.item()))
            tr_loss.append(loss.item())
            te_loss1.append(self.calLoss(self.test_x1,self.test_y1))
            te_loss2.append(self.calLoss(self.test_x2,self.test_y2))
        self.learn_curve(tr_loss,te_loss1,te_loss2)
        self.test(self.test_x1,self.test_y1)
        self.test(self.test_x2,self.test_y2)
        self.get_trainable_number()
                
if __name__=="__main__":    
    batch=15
    hidden_size=[32,64,128]
    set_length=[4,5,6]
    set_number=[3,4,5]
    for hid in hidden_size:
        for i in range(len(set_length)):
            print("set length: ",set_length[i]," set number: ",set_number[i]," hidden layer sizes: ", hid)
            newnet=Running(set_length[i],set_number[i],hid)
            newnet.importdata()
            print("--------Training---------")
            newnet.train(batch)
            print("  \n")
              
                
                
                
                