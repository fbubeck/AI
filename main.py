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
    randomForest = RandomForestRegressor.RandomForest(
        train_data, test_data)

    # Tensorflow
    tensorFlow_training = tensorFlow.train()
    tensorFlow_test = tensorFlow.test()
    tensorFlow.plot()

    # Linear Regression
    linearRegression_training = linearRegression.train()
    linearRegression_test = linearRegression.test()
    linearRegression.plot()

    # Decision Tree
    decisionTree_training = decisionTree.train()
    decisionTree_test = decisionTree.test()

    # Random Forest
    randomForest_training = randomForest.train()
    randomForest_test = randomForest.test()

    # Algorithm Comparison/Evaluation
    AlgorithmsEvaluation = Evaluation.Evaluation(train_data, test_data, tensorFlow, tensorFlow_training,
                                                 tensorFlow_test, linearRegression_training, linearRegression_test, decisionTree_training, decisionTree_test, randomForest_training, randomForest_test)
    AlgorithmsEvaluation.plot()


if __name__ == "__main__":
    main()
