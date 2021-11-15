from random import random
from algorithms import TensorFlow
from algorithms import LinearRegression
from algorithms import DecisionTree
from algorithms import RandomForestRegressor
from data import SampleData
from data import Exploration
import Evaluation
import json


def main():
    print("Starting...")
    print("")

    # read config.json
    with open('config/config.json') as file:
        config = json.load(file)

    # Get Global Parameters from config file
    n_numbers = config["GlobalParameters"]["n_numbers"]
    min_bias = config["GlobalParameters"]["min_bias"]
    max_bias = config["GlobalParameters"]["max_bias"]

    # Get Sample Data
    sampleData = SampleData.SampleData(n_numbers, min_bias, max_bias)
    train_data = sampleData.get_Data()
    test_data = sampleData.get_Data()

    # Data Exploration
    exploration = Exploration.Exploration(train_data, test_data)
    exploration.plot()

    # Creating Algorithm Objects
    tensorFlow = TensorFlow.TensorFlow(train_data, test_data)
    linearRegression = LinearRegression.LinearRegression(train_data, test_data)
    decisionTree = DecisionTree.DecisionTree(train_data, test_data)
    randomForest = RandomForestRegressor.RandomForest(train_data, test_data)

    # Tensorflow
    tensorFlow_training_duration, tensorFlow_training_error  = tensorFlow.train()
    tensorFlow_test_duration, tensorFlow_test_error, tensorFlow_y_pred  = tensorFlow.test()
    tensorFlow.plot()

    # Linear Regression
    linearRegression_training_duration, linearRegression_training_error  = linearRegression.train()
    linearRegression_test_duration, linearRegression_test_error, linearRegression_y_pred = linearRegression.test()

    # Decision Tree
    decisionTree_training_duration, decisionTree_training_error  = decisionTree.train()
    decisionTree_test_duration, decisionTree_test_error, decisionTree_y_pred = decisionTree.test()

    # Random Forest
    randomForest_training_duration, randomForest_training_error = randomForest.train()
    randomForest_test_duration, randomForest_test_error, randomForest_y_pred  = randomForest.test()

    # Algorithm Comparison/Evaluation
    AlgorithmsEvaluation = Evaluation.Evaluation(train_data, test_data, tensorFlow_training_duration, tensorFlow_training_error, tensorFlow_test_duration, tensorFlow_test_error, tensorFlow_y_pred, 
        linearRegression_training_duration, linearRegression_training_error, linearRegression_test_duration, linearRegression_test_error, linearRegression_y_pred, 
        decisionTree_training_duration, decisionTree_training_error, decisionTree_test_duration, decisionTree_test_error, decisionTree_y_pred,
        randomForest_training_duration, randomForest_training_error, randomForest_test_duration, randomForest_test_error, randomForest_y_pred)
    AlgorithmsEvaluation.plot()


if __name__ == "__main__":
    main()
