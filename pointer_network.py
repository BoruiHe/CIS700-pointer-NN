## Refercenced by https://github.com/jojonki/Pointer-Networks

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PointerNet(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, weight_size, output_size, num_layers, dict_size):
        super(PointerNet, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.weight_size = weight_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dict_size = dict_size
        self.embedding = nn.Embedding(self.dict_size,self.embed_size) 
        self.Encoder = nn.LSTM(self.embed_size,self.hidden_size,self.num_layers,batch_first=True)
        self.Decoder = nn.LSTMCell(self.hidden_size,self.hidden_size)
        
        self.W1 = nn.Linear(self.hidden_size,self.weight_size) # bias  is False
        self.W2 = nn.Linear(self.hidden_size,self.weight_size)
        self.v  = nn.Linear(self.weight_size,1)



    def forward(self,input):
        #print(input)
        input = self.embedding(input) #size: [samples_num,input_size,embed_size]
        samples_size = input.size()[0]
        #input = input.transpose(1,0)
        eh_0 = nn.Parameter(torch.zeros(self.num_layers,samples_size,self.hidden_size))
        ec_0 = nn.Parameter(torch.zeros(self.num_layers,samples_size,self.hidden_size))
        out_e, (h_n,c_n) = self.Encoder(input,(eh_0,ec_0)) # out_e size: [samples_num,input_size,hidden_size]
                                                         # h_n size: same as h_0

        out_0 = Variable(torch.zeros(samples_size,self.hidden_size))
        dh = Variable(torch.zeros(samples_size,self.hidden_size))
        dc = Variable(out_e.transpose(1,0)[-1]) # size: [samples_num,hidden_size]

        out_e = out_e.transpose(0,1)

        pred = []
        for i in range(self.output_size):
            dh,dc = self.Decoder(out_0,(dh,dc)) #dh size:[samples_num,hidden_size]
            u = F.tanh(self.W1(out_e) + self.W2(dh)) # [input_size,samples_num,weight] + [samples_num,weight]
            out_s = self.v(u).squeeze()  ## size: [input_size,samples_num]
            out_s = F.log_softmax(out_s.transpose(0,1),1)   # size: [samples_num,input_size]
            #out_s = torch.argmax(out_s,1)  # size: [samples_num]
            pred.append(out_s)

        pred = torch.stack(pred,dim=1)   # size: [samples, output_size, input_size]
        return pred









