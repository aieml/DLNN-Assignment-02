# DLNN-Assignment-02
Implementation of a simple FFNN for predicting the probability of having a heart disease 

## Instructions
- Download the Assignment folder
- In this assignment you need to train a simple FFNN model for the given dataset in "heart.csv".
- This dataset is consisted of 14 columns and 303 rows. The 1st 13 columns(0th-12th) will be considered as features and last row is consists with labels. More information is given in the next section
- All the necessary information regarding reading the dataset from the cdv file and loading into numpy arrays is given in **How to read the dataset** section below
- You need to split the dataset for training and testing, **testing size should be 20% of the whole dataset**
- Train the FFNN using ```train_data and train_target```
- Send the completed assignment with all the codes to aie.kaduwela@gmail.com

# Tasks
- Using the code given under ```Testing the Neural network and getting the accuracy``` and find the accuracy between the actual and predicted results.
- Change the FFNN architecture as you desired and try to obtain a maximum accuracy and a minimum loss.

## Dataset, features and labels

"heart.csv" contains 14 attributes, [see the original dataset @ kaggle.com](https://www.kaggle.com/ronitf/heart-disease-uci)

```
1. age- age in years
2. sex(1 = male; 0 = female)
3. cp- chest pain type
4. trestbps resting- blood pressure (in mm Hg on admission to the hospital)
5. cholserum- cholestoral in mg/dl
6. fbs- (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7. restecg- resting electrocardiographic results
8. thalach=- maximum heart rate achieved
9. exang- exercise induced angina (1 = yes; 0 = no)
10. oldpeakST- depression induced by exercise relative to rest
11. slopethe- slope of the peak exercise ST segment
12. ca- number of major vessels (0-3) colored by flourosopy
13. thal3- = normal; 6 = fixed defect; 7 = reversable defect
14. target 1 or 0
```

We can apply this dataset into a ML algorithm for future predicitions. Column 0-12 can be considered as features, all together 13 features. and the last column (13th) can be considered as labels
All the 13 features are in medical terms, therefore no need to worry about them, just consider them as features.

14th column or the target column consists of 2 labels (0,1), where **0 stands for no heart disease present** and **1 stands for heart disease is present** 

Now, you have already define feature and labels for applying the given dataset into a Deep Learning model, in this case we are going to use FFNN.

## How to read the dataset

A sample code is given in "1.0 Reading the dataset.py", you may need to install **pandas** library for python.

It can be easily done by running the below command in command prompt,

```python
pip install pandas
```

[read more about pandas](https://pandas.pydata.org/)


refer "1.0 Reading the dataset.py" 

```python
dataset=pd.read_csv('heart.csv').as_matrix()
```
This code reads the given csv file and stores it in a numpy array, in this case the size of the dataset array is 303x14.

the 1st row of the dataset, which consists the column titles will be automatically neglected when the file is beign read.

```python
data=dataset[:,0:13]
```
from 0th upto 12th columns will be assign into data (features), size will be 303x13 

```python
target=dataset[:,13]
```
13th column will be assigned as to the target, which consists the labels of 0 and 1


**You may have to use** ```python np_utils.to_categorical()``` **as shown below inorder to convert the labels into categorical labels**
```python
from keras.utils import np_utils

new_train_target=np_utils.to_categorical(train_target)
```

## Deep Feed Forward Neural Network Architecture

Refer the following NN architecture to implement the NN in Keras

| Layer         | Type          | Activation   |
| ------------- | ------------- | ------ |
| 1st Hidden Layer  | Dense, 8 Neurons  | Relu  |
| 2nd Hidden Layer  | Dense, 16 Neurons  |  Relu |
| 3rd Hidden Layer  | Dense, 8 Neurons  |  Relu |
| Output Layer  | Dense, # labels  |  Softmax |

Use appropriate loss function and use ```adam``` as the optimizer

## Testing the Neural Network and getting the accuracy

```python
predicted_targets=model.predict(test_data)

new_test_target=np_utils.to_categorical(test_target)
print(model.evaluate(new_test_target,predicted_targets))
```

```model.evaluate``` calculates the loss and the accuracy between the actual and predicted targets. Note that you have to convert the ```test_target``` into categorical target before getting the accuarcy
