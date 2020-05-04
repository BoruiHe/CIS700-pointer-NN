import numpy as np
from tqdm import tqdm

class Dataset:

    def __init__(self,data_size,seqlength=6):
        self.datasize = data_size
        self.seqlength = [np.arange(seqlength+1)[-1]]

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
        mmax = self.seqlength[-1]
        if mmax %2 == 0:
            out_size  = mmax
        else:
            out_size = mmax+1

        for i in range(len(xlist)):
            xlist[i].append(12)
            xlist[i].append(11)
        pos = len(xlist[0]) - 1

        # For ylist, the additional label is its position in input sequence.
        for i in range(len(ylist)):
            if len(ylist[i]) != out_size:
                ylist[i].append(pos-1)
            while len(ylist[i]) != out_size:
                ylist[i].append(pos)
        return xlist,ylist


    def generatedata(self):
        xlist = []
        ylist = []
        pbar = tqdm(total=self.datasize * len(self.seqlength))
        for i in self.seqlength:
            x = self.Create_lists(i)
            for j in range(len(x)):
                res = self.labels(x[j])
                if res !=[]:
                    xlist.append(x[j])
                    ylist.append(res)
                pbar.update(1)
        pbar.close()
        return self.modify(xlist,ylist)




d = Dataset(150,6)
#s = [1,2,5,3,9,8,6,9,7,2,3,6,2,1,8]
s = [2, 3, 5, 0, 9, 5]
print(d.labels(s))
xlist,ylist = d.generatedata()
print(xlist[65],ylist[65])







