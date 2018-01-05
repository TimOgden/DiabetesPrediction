from keras.models import Sequential
from keras.layers import Dense
import numpy # this tutorial doesn't use import numpy AS NP. What will that change?
# fix random seed for reproducibility
numpy.random.seed(5)

dataset = numpy.loadtxt("pima-indian-diabetes.csv", delimiter=",")
# X will be our training_data, Y will be our target_data (ex: all the factors that could determine if someone has diabetes and if they got diabetes in the past 5 years or not)
X = dataset[:,0:8] # in other words, take all values from 0-7 (inclusive) from all the rows
Y = dataset[:,8] # in other words, take each value of index 8 from all the rows

#creating model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # only question I have so far: why does the input layer have 12 neurons when there are only 8 inputs?
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # there is only one output neuron and it has a sigmoid function to switch to a 1 when above .5 and 0 when below .5
