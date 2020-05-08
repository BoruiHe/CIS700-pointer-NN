from torch import optim
from generate_data import Dataset
from torch.autograd import  Variable
from pointer_network import PointerNet
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def Default_PointerNetwork(seq_size=10, hidden_size=512, weight_size=256):
    #dict_size should be greater than the upper bound of np.random.randint() in "generate_data.py" file. 
    dict_size = 10
    #output_size should be no less than the input_size because every element might be unique.
    output_size = 2
    embed_size = 6
    num_layers=1
    pointerN = PointerNet(seq_size, embed_size, hidden_size, weight_size,output_size, num_layers, dict_size)
    return pointerN

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

def training(model, training_data_set, training_index_set, testing_data_set, testing_index_set, testing_data_d_set, testing_index_d_set, batch_size=10, n_epochs=100):
    if not(str(type(model)) == "<class 'pointer_network.PointerNet'>"):
        raise TypeError("This is not a pointer network! Check it in your Default_PointerNetwork()")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    training_loss = []
    test_loss = []
    test_loss_d = []
    for epoch in range(n_epochs):
        for i in range(0, training_data_set.size(0), batch_size):
            data = training_data_set[i:i+batch_size]
            index = training_index_set[i:i+batch_size]
            loss = getloss(data, index, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        if epoch % 50 == 0:
            print("Epoch  {} : Loss = {}".format(epoch, loss.item()))

        training_loss.append(loss.item())
        test_loss.append(getloss(testing_data_set, testing_index_set, model))
        test_loss_d.append(getloss(testing_data_d_set, testing_index_d_set, model))
    
    print("training complete")
    return training_loss, test_loss, test_loss_d

def getloss(data, index, model):
    loss_func = nn.CrossEntropyLoss()
    res = model(data).transpose(1,2).contiguous()
    res = res.view(-1,2)
    index = index.clone().detach().view(-1, 2)
    index = torch.argmax(index, 1)
    loss = loss_func(res, index)
    return loss

def testing(data, index, model):
    model.eval()
    res = model(data).transpose(1,2).contiguous()
    res = res.view(-1,2)
    res = torch.argmax(res, 1)
    index = index.clone().detach().view(-1, 2)
    index = torch.argmax(index, 1)
    shit = (res == index)
    true_counter = 0
    false_counter = 0
    for i in range(shit.size(0)):
        if shit[i] == True:
            true_counter += 1
    print("accuracy: ", 100*true_counter/(shit.size(0)), "%")

def get_trainable_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total:',total_num, 'Trainable:',trainable_num)

def learn_curve(tr,te1,te2):
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

if __name__ == "__main__":
    # seq_size = 6        #seq_size = length of single input = input_size
    batch_size = 100
    n_epoch = 100
    hidden_size = 64
    # weight_size = 32     
    shrink_factor = 0.5
    # total_samples = 400
    for weight_size in [32, 40, 48]:
        for seq_size, total_samples in zip([6, 8, 10], [1000, 700 ,400]):
            #(A, B, C, D, E, F)=(data_tensor_train, index_tensor_train, data_tensor_test, index_tensor_test, data_tensor_d, index_tensor_d)
            A, B, C, D, E, F = data_and_index_generation(total_samples, seq_size, shrink_factor)
            PointerNetwork = Default_PointerNetwork(seq_size, hidden_size, weight_size)
            print("------------------------------------------------training------------------------------------------------")
            print("number of training cases: {}, weight size: {}, sequence size: {}".format(A.size(0), weight_size, A.size(1)))
            #a, b, c = (training_loss, test_loss, test_loss_d)
            a, b, c = training(PointerNetwork, A, B, C, D, E, F, batch_size, n_epoch)
            learn_curve(a, b, c)
            print("------------------------------------------------testing------------------------------------------------")
            print("number of testing cases: {}, weight size: {}, sequence size: {}".format(C.size(0), weight_size, C.size(1)))
            testing(C, D, PointerNetwork)
            print("number of testing cases: {}, weight size: {}, sequence size: {}".format(E.size(0), weight_size, E.size(1)))
            testing(E, F, PointerNetwork)
            get_trainable_number(PointerNetwork)