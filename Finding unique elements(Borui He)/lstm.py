"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
torchvision
"""
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from generate_data import Dataset
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
# torch.manual_seed(1)    # reproducible
def data_and_index_generation(data_size, seq_size, shrink_factor):
    if shrink_factor>=1 or shrink_factor<=0:
        raise ValueError("shrink factor should be larger than 0 and less than 1! Check it in __main__.")
    elif (int)(seq_size * shrink_factor) == 0:
        raise ValueError("(seq_size * shrink factor) is too small, which leads to generating a 0 length list! Check it in __main__.")
    Data = Dataset(data_size, seq_size)
    data_list = Data.generate_data()
    index_list = Data.generate_index(data_list)
    data_tensor = Variable(torch.tensor(data_list))
    index_tensor = Variable(torch.tensor(index_list))
    
    data_split = (int)(data_size * 0.7)
    #training data generation of standard size
    data_tensor_train = data_tensor[:data_split]
    index_tensor_train = index_tensor[:data_split]
    #testing data generation of standard size(i.e. the same size of training)
    data_tensor_test = data_tensor[data_split:]
    index_tensor_test = index_tensor[data_split:]
    #testing data generation of standard size(i.e. the different size of training)
    Data_d = Dataset(data_size, (int)(seq_size*shrink_factor))
    data_list_d = Data_d.generate_data()
    index_list_d = Data_d.generate_index(data_list_d)
    data_tensor_d = Variable(torch.tensor(data_list_d))
    index_tensor_d = Variable(torch.tensor(index_list_d))

    return data_tensor_train, index_tensor_train, data_tensor_test, index_tensor_test, data_tensor_d, index_tensor_d

A, B, C, D, E, F = data_and_index_generation(2000,6,0.5)

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 6         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

x = A
y = torch.tensor(B, dtype=torch.float32)
y = index = torch.argmax(y, 2)
torch_dataset = Data.TensorDataset(x, y)

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


test_x = C   # shape (2000, 28, 28) value in range(0,1)
test_y = D    # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 32)
        self.out1 = nn.Linear(32, 12)

    def forward(self, x):
        embedding = nn.Embedding(10,6)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = embedding(x)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        out = self.out1(out)
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(100):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        # b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)
        output = output.reshape(output.shape[0],-1,2)
        output=output.reshape(-1,2) 
        b_y=b_y.reshape(-1)          # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

    if epoch % 10 == 0:
        test_output = rnn(test_x)
        test_output=test_output.reshape(test_output.shape[0],-1,2)
        test_output=test_output.reshape(-1,2)
        out_label=torch.argmax(test_output,1)



        test_yy=test_y.reshape(-1,2)
        test_yy=torch.argmax(test_yy,1)

        test_yy=test_yy.numpy()
        out_label=out_label.detach().numpy()
        right=(test_yy==out_label).astype(np.int)
        accuracy=(np.sum(right))/len(right)


        # accuracy=(torch.sum(test_yy==out_label))/(test_yy.shape[0])
        # test_output=test_output.reshape(-1,2)
        # label_out=b_y.reshape(-1)                   # (samples, time_step, input_size)
        # pred_y = torch.max(test_output, 1)[1].data.numpy()
        # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')