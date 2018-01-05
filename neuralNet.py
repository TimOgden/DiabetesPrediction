from keras.models import Sequential
from keras.layers import Dense
import numpy # this tutorial doesn't use import numpy AS NP. What will that change?
# fix random seed for reproducibility
numpy.random.seed(5)

dataset = numpy.loadtxt("pima-indian-diabetes.csv", delimiter=",")
#X will be our training_data, Y will be our target_data (ex: all the factors that could determine if someone has diabetes and if they got diabetes in the past 5 years or not)