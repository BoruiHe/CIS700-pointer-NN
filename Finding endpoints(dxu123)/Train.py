import numpy as np
import torch
import torch.nn as nn
from torch import optim
from pointer_network import PointerNet
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import  Variable
from Data_generation import Dataset


"""
To solve the problem (for macOS system), if you don't use MacOS, please comment it.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Running:

    def __init__(self,input_size=8,output_size=12,hidden_size=256,weight_size=128,samples_num=60):
        self.input_size = input_size
        self.embed_size = 64
        self.hidden_size = hidden_size
        self.weight_size = weight_size
        if self.input_size % 2== 1:
            self.output_size = output_size+1
        else:
            self.output_size = output_size
        self.num_layers = 1
        self.samples_num = samples_num
        self.dict_size = 13
        self.net = PointerNet(self.input_size+2,self.embed_size,self.hidden_size,self.weight_size,\
                              self.output_size,self.num_layers,self.dict_size)
        torch.manual_seed(10)
        if torch.cuda.is_available():
            self.net.cuda()

    def importdata(self):
        # For same size samples
        data = Dataset(self.samples_num,self.output_size,self.input_size)
        xlist,ylist = data.generatedata()

        xlist = Variable(torch.LongTensor(xlist))
        ylist = Variable(torch.LongTensor(ylist)) # the size of each label is differnt (will raise error.)
        boundary = int(0.8 * xlist.size()[0])
        self.x_train = xlist[:boundary]
        self.y_train = ylist[:boundary]
        self.xs_test = xlist[boundary:]
        self.ys_test = ylist[boundary:]
        # For different size samples
        data2 = Dataset(int(0.1*self.samples_num),self.output_size,self.output_size-2)
        xx_list,yy_list = data2.generatedata()
        self.xd_test = Variable(torch.LongTensor(xx_list))
        self.yd_test = Variable(torch.LongTensor(yy_list))

        print(self.x_train.shape,self.y_train.shape)
        print(self.xs_test.shape,self.ys_test.shape)
        print(self.xd_test.shape, self.yd_test.shape)
        #print(self.x_train[1],self.y_train[1])
        #print(self.xd_test[0],self.yd_test[0])

    def train(self,batch=40,n_epochs=100):
        self.net.train()
        optimizer = optim.Adam(self.net.parameters())
        tr_losses = [] # Traning loss
        te_losses1 = [] # testing loss(the same size)
        te_losses2 = [] # testing loss for different size
        for epoch in range(n_epochs):
            for i in range(0,self.x_train.size()[0]-batch,batch):
                x = self.x_train[i:i+batch]  # size: [batch,seq_len]
                y = self.y_train[i:i+batch]  # size: [batch, output_size]
                loss = self.calLoss(x,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print("Epoch  {} : Loss = {}".format(epoch,loss.item()))
            tr_losses.append(loss.item())
            te_losses1.append(self.calLoss(self.xs_test,self.ys_test))
            te_losses2.append(self.calLoss(self.xd_test,self.yd_test))
        self.learn_curve(tr_losses,te_losses1,te_losses2)
        self.test(self.xs_test,self.ys_test)
        self.test(self.xd_test,self.yd_test)
        self.get_trainable_number()

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

def main():
    batch = 40
    hidden_size = [64,128,256]
    seq_length = [8,12,16]
    for hid in hidden_size:
        for seq in seq_length:
            print("sequence length: ",seq,"  ", "hidden layer sizes: ",hid )
            newnet = Running(input_size=seq,output_size=seq+6,hidden_size=hid)
            newnet.importdata()
            print("--------Training---------")
            newnet.train(batch)
            print("  \n")


if __name__ == "__main__":
    main()










