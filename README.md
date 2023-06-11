# Explaining the Chronological Attribution of Greek Papyri Images
<p><img src="P.Mich.inv.%203.png"/></p>
P.Mich.inv. 3 recto, University of Michigan Library, Papyrology Collection

---

This repository comprises:
* the [PALIT dataset](https://github.com/ipavlopoulos/palit/tree/main/data/split), with images of Greek papyri from Egypt, 
* the (training of the) [fCNNr regressor](https://github.com/ipavlopoulos/palit/blob/main/code/fcnnr.ipynb) 
* the (training of the) [fCNNc classifier](https://github.com/ipavlopoulos/palit/blob/main/code/fcnnc.ipynb) 
* an exploratory analysis and our explainability study.

The abstract of our work follows:

> Greek literary papyri, which are unique witnesses of antique literature, do not usually bear a date. They are thus currently dated based on palaeographical methods, with broad approximations which often span more than a century. We created a dataset of 242 images of papyri written in "bookhand" scripts whose date can be securely assigned, and we used it to train machine and deep learning algorithms for the task of dating, showing its challenging nature. To address the data scarcity problem, we extended our dataset by segmenting each image to the respective text lines. By using the line-based version of our dataset, we trained a Convolutional Neural Network, equipped with a fragmentation-based augmentation strategy, and we achieved a mean absolute error of 54 years. The results improve further when the task is cast as a multiclass classification problem, predicting the century. Using our network, we computed and provided precise date estimations for papyri whose date is disputed or vaguely defined and we undertake an explainability-based analysis to facilitate future attribution. 
