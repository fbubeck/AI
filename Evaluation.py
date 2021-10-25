from matplotlib import pyplot as plt

class Evaluation:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def evaluateAlgorithms(self):
        __fig = plt.figure()
        plt.ylabel('Duration [in seconds]')
        plt.xlabel('Mean Squarred Error')
        plt.title('Training/Inference duration and mse of different algorithms')
        plt.scatter(tensorFlow_test[1], tensorFlow_training, s=100, c='blue', alpha=0.3)
        plt.scatter(linearRegression_test[1], linearRegression_training, s=100, c='red', alpha=0.3)
        plt.scatter(decisionTree_test[1], decisionTree_training, s=100, c='green', alpha=0.3)
        plt.scatter(randomForest_test[1], randomForest_training, s=100, c='orange', alpha=0.3)
        plt.scatter(tensorFlow_test[1], tensorFlow_test[0], s=100, c='blue', alpha=1)
        plt.scatter(linearRegression_test[1], linearRegression_test[0], s=100, c='red', alpha=1)
        plt.scatter(decisionTree_test[1], decisionTree_test[0], s=100, c='green', alpha=1)
        plt.scatter(randomForest_test[1], randomForest_test[0], s=100, c='orange', alpha=1)
        plt.legend(["TensorFlow Neural Network (Training)", "Linear Regression (Training)", "Decision Tree Regressor (Training)", "Random Forest Regressor (Training)", "TensorFlow Neural Network (Inference)", "Linear Regression (Inference)", "Decision Tree Regressor (Inference)", "Random Forest Regressor(Inference)"], loc ="upper left")
        plt.savefig('plots/Algorithms_Evaluation.png')