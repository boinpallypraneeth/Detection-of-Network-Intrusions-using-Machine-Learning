# Detection-of-Network-Intrusions-using-Machine-Learning

Project Overview: 

Intrusion detection is a fundamental part of security tools, such as adaptive security
appliances, intrusion detection systems, intrusion prevention systems, and firewalls.
Various intrusion detection techniques are used, but their performance is an issue.
Intrusion detection performance depends on accuracy, which needs to improve to
decrease false alarms and to increase the detection rate. To resolve concerns on
performance, multilayer perception, support vector machine (SVM), and other
techniques have been used in recent work. Such techniques indicate limitations and are
not efficient for use in large datasets, such as system and network data. The intrusion
detection system is used in analyzing huge traffic data; thus, an efficient classification
technique is necessary to overcome the issue. This problem is considered in this project.
Well-known machine learning techniques, namely, SVM, random forest, and extreme
learning machine are applied. These techniques are well-known because of their
capability in classification. The three most-used metrics for performance evaluation for
IDS are accuracy, True Positive Rate (TPR), and False Positive Rate (FPR). The NSL–
knowledge discovery and data mining dataset is used, which is considered a benchmark
in the evaluation of intrusion detection mechanisms. The results indicate that ELM
outperforms other approaches

A NIDS developed using ML methods usually involves following three major steps as
depicted, that is
I. Data preprocessing phase
II. Training phase
III. Testing phase.
For all the proposed solutions, the dataset is first preprocessed to transform it into the
format suitable to be used by the algorithm. This stage typically involves encoding and
normalization. Sometimes, the dataset requires cleaning in terms of removing entries
with missing data and duplicate entries, which is also performed during this phase. The
preprocessed data is then divided randomly into two portions, the training dataset, and
the testing dataset. Typically, the training dataset comprises almost 80% of the original
dataset size, and the remaining 20% forms the testing dataset. The ML algorithm is then
trained using the training dataset in the training phase. The time taken by the algorithm
in learning depends upon the size of the dataset and the complexity of the proposed
model. Normally, the training time for the models requires more training time due to
their deep and complex structure. Once the model is trained, it is tested using the testing
dataset and evaluated based on the predictions it made. In the case of NIDS models, the
network traffic instance will be predicted to belong to either benign (normal) or attack
class.

A NIDS developed using ML methods usually involves following three major steps as
depicted, that is
I. Data preprocessing phase
II. Training phase
III. Testing phase.
For all the proposed solutions, the dataset is first preprocessed to transform it into the
format suitable to be used by the algorithm. This stage typically involves encoding and
normalization. Sometimes, the dataset requires cleaning in terms of removing entries
with missing data and duplicate entries, which is also performed during this phase. The
preprocessed data is then divided randomly into two portions, the training dataset, and
the testing dataset. Typically, the training dataset comprises almost 80% of the original
dataset size, and the remaining 20% forms the testing dataset. The ML algorithm is then
trained using the training dataset in the training phase. The time taken by the algorithm
in learning depends upon the size of the dataset and the complexity of the proposed
model. Normally, the training time for the models requires more training time due to
their deep and complex structure. Once the model is trained, it is tested using the testing
dataset and evaluated based on the predictions it made. In the case of NIDS models, the
network traffic instance will be predicted to belong to either benign (normal) or attack
class.

Modules Used
Dataset
Dataset selection for experimentation is a significant task because the performance of
the system is based on the correctness of a dataset. The more accurate the data, the
greater the effectiveness of the system. The dataset can be collected by numerous
means, such as
1) sanitized dataset,
2) simulated dataset,
3) testbed dataset,
4) standard dataset.

Additionally, different types of traffic are required to model various network attacks,
which is complex and costly. To overcome these difficulties, the NSL–KDD dataset is
used to validate the proposed system for intrusion detection.

Pre-Processing:
The classifier is unable to process the raw dataset because of some of its symbolic
features. Thus, pre-processing is essential, in which non-numeric or symbolic features
are eliminated or replaced, because they do not indicate vital participation in intrusion
detection. However, this process generates overhead including more training time; the
classifier’s architecture becomes complex and wastes memory and computing
resources. Therefore, the non-numeric features are excluded from the raw dataset for
improved performance of intrusion detection systems.


Classification:
The present proposed method is based mainly on two main phases, the first phase is to
detect and pre-process the eye images using the image processing technique and the
second phase is to build a classification model that will be able to classify whether the
eye is opened or closed and then start an alarm accordingly.
Support Vector Machine:
Implementation of the SVM model in the proposed system. The kernel function uses
squared Euclidean distance between two numeric vectors and maps input data to a high
dimensional space to optimally separate the given data into their respective attack
classes. Therefore, kernel RBF is particularly effective in separating sets of data that
share complex boundaries. In our study, all the simulations have been conducted using
the freely available Lib SVM package.


Random Forest
RFs are ensemble classifiers, which are used for classification and regression analysis
of the intrusion detection data. RF works by creating various decision trees in the
training phase and output class labels that have the majority vote. RF attains high
classification accuracy and can handle outliers and noise in the data. RF is used in this
work because it is less susceptible to over-fitting and it has previously shown good
classification results.


Extreme Learning Machine:
ELM is another name for single or multiple hidden layer feed-forward neural networks.
ELM can be used to solve various classification, clustering, regression, and feature
engineering problems. This learning algorithm involves an input layer, one or multiple
hidden layers, and the output layer. In the traditional neural networks, the tasks of
adjustment of the input and hidden layer weights are very computationally expensive
and time-consuming because it requires multiple rounds to converge. To overcome this
problem, Huang et al. proposed an SLFN by arbitrarily selecting input weights and
hidden layer biases to minimize the training time.


EVALUATION METRICS
This section explains the most commonly used evaluation metrics for measuring the
performance of ML and methods for IDS. All the evaluation metrics are based on the
different attributes used in the Confusion Matrix, which is a two-dimensional matrix
providing information about the Actual and Predicted class and includes;
i. True Positive (TP): The data instances are correctly predicted as an Attack by
the classifier.
ii. False Negative (FN): The data instances were wrongly predicted as Normal
instances.
iii. False Positive (FP): The data instances were wrongly classified as an Attack.
iv. True Negative (TN): The instances are correctly classified as Normal
instances.
The diagonal of the confusion matrix denotes the correct predictions while nondiagonal
elements are the wrong predictions of a certain classifier. Further, the different
evaluation metrics used are,

* Precision: It is the ratio of correctly predicted Attacks to all the samples predicted
as Attacks.
Precision= TP/TP+FP
* Recall: It is a ratio of all samples correctly classified as Attacks to all the samples
that are Attacks. It is also called a Detection Rate.
Recall=Detection Rate= TP/TP+FP
* False alarm rate: It is also called the false positive rate and is defined as the ratio
of wrongly predicted Attack samples to all the Normal samples.
False Alarm Rate =FP/FP + TN
* True negative rate: It is defined as the ratio of the number of correctly classified
Normal samples to all the Normal samples.
True Negative Rate =TN/TN+FP
* Accuracy: It is the ratio of correctly classified instances to the total number of
instances. It is also called Detection Accuracy and is a useful performance
measure only when a dataset is balanced.
Accuracy =TP+TN/TP+TN+FP+FN
* F-Measure: It is defined as the harmonic mean of the Precision and Recall. In
other words, it is a statistical technique for examining the accuracy of a system by
considering both the precision and recall of the system.
F Measure = 2 (Precision × Recall)/(Precision + Recall)

CONCLUSION
Intrusion detection and prevention are essential to current and future networks and
information systems because our daily activities are heavily dependent on them.
Furthermore, future challenges will become more daunting because of the Internet of
Things. In this respect, intrusion detection systems have been important in the last few
decades. Several techniques have been used in intrusion detection systems, but machine
learning techniques are common in recent literature. Additionally, different machine
learning techniques have been used, but some techniques are more suitable for
analyzing huge data for intrusion detection of network and information systems. To
address this problem, different machine learning techniques, namely, SVM, RF, and
ELM are investigated and compared in this work. ELM outperforms other approaches
in accuracy, precision, and recall on the full data samples that comprise 65,535 records
of activities containing normal and intrusive activities. Furthermore, the SVM indicated
better results than other datasets in half of the data samples and 1/4 of the data samples.
Therefore, ELM is a suitable technique for intrusion detection systems that are designed
to analyze a huge amount of data. In the future, ELM will be explored further to
investigate its performance in feature selection and feature transformation techniques.
