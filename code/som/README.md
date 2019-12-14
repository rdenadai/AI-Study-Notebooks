## **Self-Organizing Map**

Self-Organizing Map (SOM) is a unsupervised algorithm which maps a lower dimension map to a higher dimension data.

One propurse could be clustering or even dimensionality reduction.

![SOM](https://github.com/rdenadai/AI-Study-Notebooks/blob/master/images/som.gif?raw=true)

The map is build using a grid with nxn dimensions and each unit (neuron) of the map contains a value(s) with the dimension of the data. This way we are able to map a high dimension to a lower nxn dimension.

The self-organizing structure is given by moving each unit in the direction of the data points using some distance metric (like euclidian distance as is the most common).

Each epoch (or iteration), the points in the map are moving towards the clusters inside the data.

At the end, one should see the data topology using a U-Matrix.

All of these are implemented in this notebook.

    cuda:0


### Implementation

 - The implementation uses cython and python multiprocessing to improve the performance in calculate SOM.

 - The distance metric used is the euclidian distance.

 - This implementation uses a decay rate to lower the learning rate and neighbour units after some define number of iterations (given by the iter_decay).

 - There's still no implementation of clustering... yet...

### **Toy datasets provided by scikit-learn**

To initialy test and visualize what a SOM do, we are going to use toy datasets provided by scikit-learn.

Later on this notebook is explored a way to create a latent space of the MNIST dataset using Autoencoders and the apply SOM on the result.

#### 3D Blobs

    epoch [25/200] <=> Running time: 7.066114664077759
    epoch [50/200] <=> Running time: 13.8524649143219
    epoch [75/200] <=> Running time: 18.966942071914673
    epoch [100/200] <=> Running time: 23.207172393798828
    epoch [125/200] <=> Running time: 26.59671688079834
    epoch [150/200] <=> Running time: 29.89101791381836
    epoch [175/200] <=> Running time: 32.80154633522034
    epoch [199/200] <=> Running time: 36.626179218292236



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_9_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_10_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_11_0.png)


#### 2D Blobs

    epoch [25/200] <=> Running time: 4.368779182434082
    epoch [50/200] <=> Running time: 8.014280319213867
    epoch [75/200] <=> Running time: 10.712982892990112
    epoch [100/200] <=> Running time: 12.925299882888794
    epoch [125/200] <=> Running time: 14.592494010925293
    epoch [150/200] <=> Running time: 16.27384376525879
    epoch [175/200] <=> Running time: 17.880710124969482
    epoch [199/200] <=> Running time: 19.7550106048584



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


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_54_0.png)


#### Autoencoder network

#### Training step

    [1/51] loss: 0.2212930620, time: 16.54
    [2/51] loss: 0.1497398019, time: 16.07
    [3/51] loss: 0.1272169948, time: 16.68
    [4/51] loss: 0.1251637340, time: 16.43
    [5/51] loss: 0.0879302770, time: 16.39
    [6/51] loss: 0.1058123261, time: 16.13
    [7/51] loss: 0.1018903702, time: 16.48
    [8/51] loss: 0.0895036981, time: 15.97
    [9/51] loss: 0.0815045908, time: 16.25
    [10/51] loss: 0.0910560489, time: 16.93
    [11/51] loss: 0.1082131714, time: 15.86
    [12/51] loss: 0.1175806150, time: 15.41
    [13/51] loss: 0.1083476618, time: 15.45
    [14/51] loss: 0.0968725160, time: 15.77
    [15/51] loss: 0.1158992872, time: 15.46
    [16/51] loss: 0.0937166214, time: 15.59
    [17/51] loss: 0.0869348273, time: 15.73
    [18/51] loss: 0.0821733996, time: 15.88
    [19/51] loss: 0.0995057076, time: 16.75
    [20/51] loss: 0.0742746666, time: 16.74
    [21/51] loss: 0.0941771269, time: 16.18
    [22/51] loss: 0.0744719505, time: 16.46
    [23/51] loss: 0.0790197328, time: 16.40
    [24/51] loss: 0.1034879759, time: 16.32
    [25/51] loss: 0.0804327875, time: 16.49
    [26/51] loss: 0.0879556686, time: 16.01
    [27/51] loss: 0.0707675368, time: 16.07
    [28/51] loss: 0.0851430595, time: 15.80
    [29/51] loss: 0.1001406685, time: 15.85
    [30/51] loss: 0.0902497023, time: 15.92
    [31/51] loss: 0.0818073153, time: 15.80
    [32/51] loss: 0.0612869859, time: 16.41
    [33/51] loss: 0.0900170580, time: 15.98
    [34/51] loss: 0.0633055940, time: 15.54
    [35/51] loss: 0.0836695880, time: 15.45
    [36/51] loss: 0.0871250182, time: 15.83
    [37/51] loss: 0.0861932188, time: 15.51
    [38/51] loss: 0.0700632483, time: 16.21
    [39/51] loss: 0.0904704332, time: 16.46
    [40/51] loss: 0.0577094220, time: 16.51
    [41/51] loss: 0.0661170557, time: 16.41
    [42/51] loss: 0.0869717970, time: 16.74
    [43/51] loss: 0.0640795603, time: 16.89
    [44/51] loss: 0.0521222800, time: 16.74
    [45/51] loss: 0.0892381892, time: 16.20
    [46/51] loss: 0.0754079074, time: 16.14
    [47/51] loss: 0.0713761449, time: 16.34
    [48/51] loss: 0.0755914599, time: 16.89
    [49/51] loss: 0.0657814592, time: 16.58
    [50/51] loss: 0.0926574618, time: 16.75
    [51/51] loss: 0.0809435397, time: 16.27


#### Results

The first line is the ground truth values and the line bellow is the decoded version... each column represents an epoch!



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_60_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_60_1.png)


Bellow i'm just loading the best saved model!

Now let's generated the "predicted" version from our test data and also, generate the latent space and reserve that to be used in the SOM!

The first 3 lines are the ground truth of tests and the rest is de decoded version... Looks like it do 2 mistakes in this small data visualization but in general looks fine!


![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_66_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_66_1.png)


Let's use **t-SNE** to do a dimensionality reduction before pass to the SOM.

    epoch [25/100] <=> Running time: 157.77867650985718
    epoch [50/100] <=> Running time: 271.88222217559814
    epoch [75/100] <=> Running time: 360.770840883255
    epoch [99/100] <=> Running time: 427.18655252456665



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_69_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_70_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_71_0.png)


Results are not very good... but we can see some clusters in the reduced data... let's run the SOM in the original 10 dim latent space generated by the Autoencoder and see if it performs better!

    epoch [25/200] <=> Running time: 291.0855915546417
    epoch [50/200] <=> Running time: 511.19567370414734
    epoch [75/200] <=> Running time: 668.1621434688568
    epoch [100/200] <=> Running time: 781.1210885047913
    epoch [125/200] <=> Running time: 863.6010956764221
    epoch [150/200] <=> Running time: 947.708149433136
    epoch [175/200] <=> Running time: 1031.3704252243042
    epoch [199/200] <=> Running time: 1109.147782087326



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_74_0.png)



![png](7.%20SOM%20%28Kohonen%20MAP%29_files/7.%20SOM%20%28Kohonen%20MAP%29_75_0.png)


Not much better than the example using t-SNE. We could see some clusters, but as the same case as PCA the Autoencoder could not generate the lastent space that fully separate the data... Well in this case, perhaps the Autoencoder needs to be tunned or reformulate in a way that could mimic in a way whats t-SNE do. 
