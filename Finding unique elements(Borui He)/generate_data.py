import numpy as np

class Dataset:
    def __init__(self, data_size=10, seq_size=6):
        self.datasize = data_size
        self.seq_size = seq_size

    def generate_data(self):
        data_list = []
        for i in range(self.datasize):
            data_list.append(np.random.randint(0,10,size = self.seq_size).tolist())
        return data_list

    def generate_index(self, data_list):
        index_list = []
        for i in range(self.datasize):
            sub_index_list= []
            for j in range(len(data_list[i])):
                copy_data_list = data_list[i].copy()
                copy_data_list.remove(copy_data_list[j])
                # print("\nj:",j)
                # print("data_list[i]:", data_list[i])
                # print("data_list[i][j]:", data_list[i][j])
                # print("c_d_l:", copy_data_list)
                # print("res:", data_list[i][j] in copy_data_list)
                if not(data_list[i][j] in copy_data_list):
                    sub_index_list.append([0,1])    #if unique, append [0,1]
                else:
                    sub_index_list.append([1,0])    #otherwise, append [1,0]
                    # print("sub_i_l:", sub_index_list)
            index_list.append(sub_index_list)
            # print("index_l:", index_list)
        return index_list

if __name__ == "__main__":
    b = Dataset().generate_data()
    print("b:\n",b)
    c = Dataset().generate_index(b)
    print("c:\n",c)
    print(type(b), type(c))
