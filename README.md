# Explaining the Chronological Attribution of Greek Papyri Images

In this repository is the PALIT dataset, comprising images of Greek papyri from Egypt, and the fCNN regressor and classifier, with which we estimated more precise chronology for papyri images whose date ranged between centuries. The abstract of our work follows:

> Greek literary papyri, which are unique witnesses of antique literature, do not usually bear a date. They are thus currently dated based on palaeographical methods, with broad approximations which often span more than a century. We created a dataset of 242 images of papyri written in "bookhand" scripts whose date can be securely assigned, and we used it to train machine and deep learning algorithms for the task of dating, showing its challenging nature. To address the data scarcity problem, we extended our dataset by segmenting each image to the respective text lines. By using the line-based version of our dataset, we trained a Convolutional Neural Network, equipped with a fragmentation-based augmentation strategy, and we achieved a mean absolute error of 54 years. The results improve further when the task is cast as a multiclass classification problem, predicting the century. Using our network, we computed and provided precise date estimations for papyri whose date is disputed or vaguely defined and we undertake an explainability-based analysis to facilitate future attribution. 

![P.Mich.inv. 3 recto, University of Michigan Library, Papyrology Collection](P.Mich.inv. 3.png)
P.Mich.inv. 3 recto, University of Michigan Library, Papyrology Collection

You may find the data [here](https://github.com/ipavlopoulos/palit/tree/main/data/split), the training of our fCNN classifier [here](https://github.com/ipavlopoulos/palit/blob/main/code/fcnnc.ipynb), the training of our fCNN regressor [here](https://github.com/ipavlopoulos/palit/blob/main/code/fcnnr.ipynb).
