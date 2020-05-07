import numpy as np
from tqdm import tqdm

class Dataset:

    def __init__(self,data_size,maximum_output,seqlength=6):
        self.datasize = data_size
        self.seqlength = np.arange(seqlength+1)[5:]
        self.maxoutput = maximum_output

    def Create_lists(self,n):
        dataset = []
        np.random.seed(10)
        for i in range(self.datasize):
            dataset.append(np.random.randint(0,10,size = n).tolist())
        return dataset

    def labels(self,seq):
        ylist = []
        i = 0
        while i<len(seq):
            if seq[i] > 5:
                ylist.append(i)
                i += 1
                while(i<len(seq) and seq[i]>5):
                    i+=1
                ylist.append(i-1)
            i += 1
        return ylist


    def modify(self,xlist,ylist):
        ## let "<pad>" = 11 ,"<EOS>" = 12 (encoding)
        if self.maxoutput %2 == 0:
            out_size  = self.maxoutput
        else:
            out_size = self.maxoutput+1
        # To get the same size out each output, we should add some other labels for both xlist and ylist
        for i in range(len(xlist)):
            xlist[i].append(12)
            xlist[i].append(11)
            pos = len(xlist[i])-1
            while len(xlist[i]) != out_size:
                xlist[i].append(11)
            if len(ylist[i]) != out_size:
                ylist[i].append(pos-1)
            while len(ylist[i]) != out_size:
                ylist[i].append(pos)
        return self.shuffle_data(xlist,ylist)

    def shuffle_data(self,xlist,ylist):
        c = list(zip(xlist,ylist))
        np.random.seed(10)
        np.random.shuffle(c)
        xlist,ylist = zip(*c)
        return xlist,ylist


    def generatedata(self):
        xlist = []
        ylist = []
        pbar = tqdm(total=self.datasize * len(self.seqlength))
        for i in self.seqlength:
            x = self.Create_lists(i)
            for j in range(len(x)):
                res = self.labels(x[j])
                #if res !=[]:
                xlist.append(x[j])
                ylist.append(res)
                pbar.update(1)
        pbar.close()
        return self.modify(xlist,ylist)


"""
d = Dataset(15,8,6)
#s = [1,2,5,3,9,8,6,9,7,2,3,6,2,1,8]
#s = [2, 3, 5, 0, 9, 5]
#print(d.labels(s))
xlist,ylist = d.generatedata()
print(xlist[65],ylist[65])
"""









