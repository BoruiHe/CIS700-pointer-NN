# Evaluate different applications with Pointer Networks

We used Pointer Network to do three different tasks. Three tasks are finding unique elements in a given list, finding all endpoints in a sequence and set operations. Pointer Network demonstrated an excellent performance in finding all endpoints while it seems worked not well in finding unique elements and set operations. In the results part, pointer networks with larger weight size or layer size have relatively better performance. It indicates that there is a positive correlation between such two configurations and networks’ learning efficiency. However it also seems to have some upper bound for learning efficiency shown in Jiaming’s cases. 




## Prerequisites
You need to install some packages like `numpy`, `pytorch`, `tqdm` and `matplotlib`. 

The version of Python is 3.7.4.

The version of each packages:

- `numpy`: version 1.18.1

- `torch`: version  1.5.0

- `tqdm`: version 4.36.1

- `matplotlib` : version 3.1.3

  

If you use macOS system, you must add following codes at the begining of  `train.py` file :

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```



## Running the test

We can't always get the same results because of its random initialization of the parameters in the network. Although we have set a seed for random process, we cannot let each running get the same result due to differenct kinds of CPU or GPU. 



#### For "Finding unique element" task(Borui):

1. All the necessary code is under the same directory Finding unique elements`(Borui He)/`. Do not use any file under parent directory instead.
2. Download ZIP file and unzip the directory Finding unique elements `(Borui He)/` to your desktop.
3. Open the "main.py" file and run it.



#### For "Finding all endpoints in a sequence" task(Diliao):

1. Download the ZIP file and unzip it. The extracted directory is named `CIS700-pointer-NN-master`. 
2. Change directory to `./Finding endpoints(dxu123)/`. 
3. Just run the file `train.py`. 



#### For "Set operations" task(Jiaming):

1. Download the ZIP file and unzip it. The extracted directory is named `CIS700-pointer-NN-master`. 
2. Change directory to `./Set operations(jzhan252)/`. 
3. Just run the file `training.py`. 
