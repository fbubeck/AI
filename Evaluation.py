from matplotlib import pyplot as plt


class Evaluation:
    def __init__(self, train_data, test_data, tensorFlow, tensorFlow_training,
                 tensorFlow_test, linearRegression_training, linearRegression_test, decisionTree_training, decisionTree_test, randomForest_training, randomForest_test):
        self.train_data = train_data
        self.test_data = test_data
        self.tensorFlow = tensorFlow
        self.tensorFlow_training = tensorFlow_training
        self.tensorFlow_test = tensorFlow_test
        self.linearRegression_training = linearRegression_training
        self.linearRegression_test = linearRegression_test
        self.decisionTree_training = decisionTree_training
        self.decisionTree_test = decisionTree_test
        self.randomForest_training = randomForest_training
        self.randomForest_test = randomForest_test

    def plot(self):
        px = 1/plt.rcParams['figure.dpi']  
        __fig = plt.figure(figsize=(800*px, 600*px))
        plt.ylabel('Duration [in seconds]')
        plt.xlabel('Mean Squarred Error')
        plt.title('Training/Inference duration and mse of different ML-Algorithms')
        plt.scatter(self.tensorFlow_test[1], self.tensorFlow_training, s=100, c='blue', alpha=1)
        plt.scatter(self.tensorFlow_test[1], self.tensorFlow_test[0], s=100, c='blue', alpha=0.3)
        plt.scatter(self.linearRegression_test[1], self.linearRegression_training, s=100, c='red', alpha=1)
        plt.scatter(self.linearRegression_test[1], self.linearRegression_test[0], s=100, c='red', alpha=0.3)
        plt.scatter(self.decisionTree_test[1], self.decisionTree_training, s=100, c='green', alpha=1)
        plt.scatter(self.decisionTree_test[1], self.decisionTree_test[0], s=100, c='green', alpha=0.3)
        plt.scatter(self.randomForest_test[1], self.randomForest_training, s=100, c='orange', alpha=1)
        plt.scatter(self.randomForest_test[1], self.randomForest_test[0], s=100, c='orange', alpha=0.3)
        plt.legend(["TensorFlow Neural Network (Training)", "TensorFlow Neural Network (Inference)", "Linear Regression (Training)", "Linear Regression (Inference)", "Decision Tree Regressor (Training)", "Decision Tree Regressor (Inference)", "Random Forest Regressor (Training)", "Random Forest Regressor(Inference)"], loc="upper right")
        plt.savefig('plots/Algorithms_Evaluation.png')
        plt.show()
        print("Evaluation Plot saved...")
        print("")
