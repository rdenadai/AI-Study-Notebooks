## **Self-Organizing Map**

Self-Organizing Map (SOM) is a unsupervised algorithm which maps a lower dimension map to a higher dimension data.

One propurse could be clustering or even dimensionality reduction.

![SOM](https://github.com/rdenadai/AI-Study-Notebooks/tree/master/images/som.gif)

The map is build using a grid with nxn dimensions and each unit (neuron) of the map contains a value(s) with the dimension of the data. This way we are able to map a high dimension to a lower nxn dimension.

The self-organizing structure is given by moving each unit in the direction of the data points using some distance metric (like euclidian distance as is the most common).

Each epoch (or iteration), the points in the map are moving towards the clusters inside the data.

At the end, one should see the data topology using a U-Matrix.

All of these are implemented in this notebook.

    Using cpu...


### Implementation

 - The implementation uses cython and python multiprocessing to improve the performance in calculate SOM.

 - The distance metric used is the euclidian distance.

 - This implementation uses a decay rate to lower the learning rate and neighbour units after some define number of iterations (given by the iter_decay).

 - There's still no implementation of clustering... yet...

### **Toy datasets provided by scikit-learn**

To initialy test and visualize what a SOM do, we are going to use toy datasets provided by scikit-learn.

Later on this notebook is explored a way to create a latent space of the MNIST dataset using Autoencoders and the apply SOM on the result.

#### 3D Blobs

    epoch [25/200] <=> Running time: 10.0800302028656
    epoch [50/200] <=> Running time: 19.663402795791626
    epoch [75/200] <=> Running time: 27.182862281799316
    epoch [100/200] <=> Running time: 33.35936117172241
    epoch [125/200] <=> Running time: 37.36079931259155
    epoch [150/200] <=> Running time: 41.348073959350586
    epoch [175/200] <=> Running time: 43.851285219192505
    epoch [199/200] <=> Running time: 46.130627155303955



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_9_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_10_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_11_0.png)


#### 2D Blobs

    epoch [25/200] <=> Running time: 6.300544500350952
    epoch [50/200] <=> Running time: 11.740918159484863
    epoch [75/200] <=> Running time: 15.853654384613037
    epoch [100/200] <=> Running time: 18.906008005142212
    epoch [125/200] <=> Running time: 20.929914236068726
    epoch [150/200] <=> Running time: 23.021363973617554
    epoch [175/200] <=> Running time: 24.241161108016968
    epoch [199/200] <=> Running time: 25.426844835281372



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_13_1.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_14_0.png)


#### Two Moons (2D)


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_17_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_18_0.png)


#### Another 2D Blobs


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_21_0.png)


#### Another 3D Blobs


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_24_0.png)


#### Rings (2D)


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_27_0.png)


#### Clusters of Blobs


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_30_0.png)


#### 3D Clusters of Blobs


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_33_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_34_0.png)


## **Self-organizing map on IRIS**


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_37_0.png)


### **Self-organizing map on MNIST**


Loading MNIST from scikit-learn to test out against the Self-Organizing Map.


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_39_0.png)


#### MNIST : PCA (2D)


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_42_0.png)


#### MNIST : PCA (3D)


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_45_0.png)


### MNIST : t-SNE (2D)


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_48_0.png)


### MNIST : t-SNE (3D)


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_51_0.png)


### **Autoencoder**

Taken from and modified version from : [Autoencoder in Pytorch](https://reyhaneaskari.github.io/AE.htm)

The problem with the above techniques is that it works fine with a low dimensional image, like the toy MNIST provided by scikit-learn.

If one try with the MNIST dataset provided by pytorch, well... its a little differente history.

    0it [00:00, ?it/s]

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz


    9920512it [00:06, 1588962.93it/s]                            


    Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw


      0%|          | 0/28881 [00:00<?, ?it/s]

    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz


    32768it [00:00, 137448.71it/s]           
      0%|          | 0/1648877 [00:00<?, ?it/s]

    Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz


    1654784it [00:00, 2260738.83it/s]                            
      0%|          | 0/4542 [00:00<?, ?it/s]

    Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz


    8192it [00:00, 54156.22it/s]            
    0it [00:00, ?it/s]

    Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw
    Processing...
    Done!
    Downloading https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz to ./data/QMNIST/raw/qmnist-test-images-idx3-ubyte.gz


    9748480it [00:00, 25438372.78it/s]         
    0it [00:00, ?it/s]

    Downloading https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz to ./data/QMNIST/raw/qmnist-test-labels-idx2-int.gz


    532480it [00:00, 2204021.03it/s]          


    Processing...



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_54_0.png)


#### Autoencoder network

#### Training step

    [1/21] loss: 0.1966524720, time: 23.61
    [2/21] loss: 0.1576950103, time: 22.83
    [3/21] loss: 0.1423753500, time: 23.08
    [4/21] loss: 0.1070745364, time: 23.18
    [5/21] loss: 0.1161808446, time: 22.94
    [6/21] loss: 0.1033642441, time: 22.16
    [7/21] loss: 0.1068933606, time: 22.94
    [8/21] loss: 0.1079141647, time: 22.54
    [9/21] loss: 0.0915218294, time: 23.32
    [10/21] loss: 0.0962715074, time: 22.60
    [11/21] loss: 0.0920675248, time: 22.45
    [12/21] loss: 0.0759960860, time: 23.04
    [13/21] loss: 0.1056505293, time: 24.24
    [14/21] loss: 0.0813286826, time: 22.70
    [15/21] loss: 0.0814369097, time: 22.88
    [16/21] loss: 0.0928150266, time: 23.07
    [17/21] loss: 0.0717718676, time: 22.73
    [18/21] loss: 0.0854445398, time: 22.25
    [19/21] loss: 0.0697054118, time: 23.17
    [20/21] loss: 0.0856528282, time: 22.19
    [21/21] loss: 0.0637218356, time: 23.09


#### Results

The first line is the ground truth values and the line bellow is the decoded version... each column represents an epoch!



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_60_0.png)


Bellow i'm just loading the best saved model!

Now let's generated the "predicted" version from our test data and also, generate the latent space and reserve that to be used in the SOM!

The first 3 lines are the ground truth of tests and the rest is de decoded version... Looks like it do 2 mistakes in this small data visualization but in general looks fine!


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_66_0.png)


Let's use **t-SNE** to do a dimensionality reduction before pass to the SOM.

    epoch [25/100] <=> Running time: 136.45961928367615
    epoch [50/100] <=> Running time: 235.25226640701294
    epoch [75/100] <=> Running time: 309.74252247810364
    epoch [99/100] <=> Running time: 358.7991991043091



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_69_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_70_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_71_0.png)


Results are not very good... but we can see some clusters in the reduced data... let's run the SOM in the original 10 dim latent space generated by the Autoencoder and see if it performs better!

    epoch [25/200] <=> Running time: 170.36671614646912
    epoch [50/200] <=> Running time: 311.32559609413147
    epoch [75/200] <=> Running time: 420.83449244499207
    epoch [100/200] <=> Running time: 500.99005818367004
    epoch [125/200] <=> Running time: 553.0462336540222
    epoch [150/200] <=> Running time: 604.6043317317963
    epoch [175/200] <=> Running time: 635.0623729228973
    epoch [199/200] <=> Running time: 664.0316405296326



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_74_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_75_0.png)


Not much better than the example using t-SNE. We could see some clusters, but as the same case as PCA the Autoencoder could not generate the lastent space that fully separate the data... Well in this case, perhaps the Autoencoder needs to be tunned or reformulate in a way that could mimic in a way whats t-SNE do. 
