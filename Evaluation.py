from matplotlib import pyplot as plt


class Evaluation:
    def __init__(self, train_data, test_data, tensorFlow_training_duration, tensorFlow_training_mse, tensorFlow_test_duration, tensorFlow_test_mse, tensorFlow_y_pred, 
        linearRegression_training_duration, linearRegression_training_mse, linearRegression_test_duration, linearRegression_test_mse, linearRegression_y_pred, 
        decisionTree_training_duration, decisionTree_training_mse, decisionTree_test_duration, decisionTree_test_mse, decisionTree_y_pred,
        randomForest_training_duration, randomForest_training_mse, randomForest_test_duration, randomForest_test_mse, randomForest_y_pred):

        self.train_data = train_data
        self.test_data = test_data
        self.xs_test = test_data[0]
        self.ys_test = test_data[1]
        self.tensorFlow_training_duration = tensorFlow_training_duration
        self.tensorFlow_training_mse = tensorFlow_training_mse
        self.tensorFlow_test_duration = tensorFlow_test_duration
        self.tensorFlow_test_mse = tensorFlow_test_mse
        self.tensorFlow_y_pred = tensorFlow_y_pred
        self.linearRegression_training_duration = linearRegression_training_duration
        self.linearRegression_training_mse = linearRegression_training_mse
        self.linearRegression_test_duration = linearRegression_test_duration
        self.linearRegression_test_mse = linearRegression_test_mse
        self.linearRegression_y_pred = linearRegression_y_pred
        self.decisionTree_training_duration = decisionTree_training_duration
        self.decisionTree_training_mse = decisionTree_training_mse
        self.decisionTree_test_duration = decisionTree_test_duration
        self.decisionTree_test_mse = decisionTree_test_mse
        self.decisionTree_y_pred = decisionTree_y_pred
        self.randomForest_training_duration = randomForest_training_duration
        self.randomForest_training_mse = randomForest_training_mse
        self.randomForest_test_duration = randomForest_test_duration
        self.randomForest_test_mse = randomForest_test_mse
        self.randomForest_y_pred = randomForest_y_pred

    def plot(self):
        px = 1/plt.rcParams['figure.dpi']

        fig = plt.figure(figsize=(1200*px, 800*px))
        fig.suptitle('Model Comparison')
        axs1 = fig.add_subplot(221)
        axs2 = fig.add_subplot(222)
        axs3 = fig.add_subplot(223)
        axs4 = fig.add_subplot(224)
        axs1.scatter(self.xs_test, self.ys_test, color='b', s=1, label="Test Data", alpha=0.5)
        axs1.scatter(self.xs_test, self.tensorFlow_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
        axs2.scatter(self.xs_test, self.ys_test, color='b', s=1, label="Test Data", alpha=0.5)
        axs2.scatter(self.xs_test, self.linearRegression_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
        axs3.scatter(self.xs_test, self.ys_test, color='b', s=1, label="Test Data", alpha=0.5)
        axs3.scatter(self.xs_test, self.decisionTree_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
        axs4.scatter(self.xs_test, self.ys_test, color='b', s=1, label="Test Data", alpha=0.5)
        axs4.scatter(self.xs_test, self.randomForest_y_pred, color='r', s=1, label="Predicted Data", alpha=0.5)
        axs1.title.set_text('TensorFlow Model')
        axs2.title.set_text('Linear Regression Model')
        axs3.title.set_text('Decision Tree Model')
        axs4.title.set_text('Random Forest Model')
        axs1.legend(loc="upper left", markerscale=10, scatterpoints=1)
        axs2.legend(loc="upper left", markerscale=10, scatterpoints=1)
        axs3.legend(loc="upper left", markerscale=10, scatterpoints=1)
        axs4.legend(loc="upper left", markerscale=10, scatterpoints=1)
        axs1.text(.95, 0.05, ("Error: " + str(self.tensorFlow_test_mse)), ha='right', va='bottom', transform=axs1.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
        axs2.text(.95, 0.05, ("Error: " + str(self.linearRegression_test_mse)), ha='right', va='bottom', transform=axs2.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
        axs3.text(.95, 0.05, ("Error: " + str(self.decisionTree_test_mse)), ha='right', va='bottom', transform=axs3.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
        axs4.text(.95, 0.05, ("Error: " + str(self.randomForest_test_mse)), ha='right', va='bottom', transform=axs4.transAxes, bbox=dict(boxstyle = "square", facecolor = "white", alpha = 0.5))
        plt.savefig('plots/Algorithms_Model_Comparison.png')
        #plt.show()



        __fig = plt.figure(figsize=(1200*px, 800*px))
        plt.ylabel('Duration [in seconds]')
        plt.xlabel('Error')
        plt.title('Training/Inference duration and error of different ML-Algorithms')
        plt.scatter(self.tensorFlow_training_mse, self.tensorFlow_training_duration, s=120, c='blue', alpha=0.6)
        plt.scatter(self.tensorFlow_test_mse, self.tensorFlow_test_duration, s=120, c='blue', alpha=0.6, marker='v')
        plt.scatter(self.linearRegression_training_mse, self.linearRegression_training_duration, s=120, c='red', alpha=0.6)
        plt.scatter(self.linearRegression_test_mse, self.linearRegression_test_duration, s=120, c='red', alpha=0.6, marker='v')
        plt.scatter(self.decisionTree_training_mse, self.decisionTree_training_duration, s=120, c='green', alpha=0.6)
        plt.scatter(self.decisionTree_test_mse, self.decisionTree_test_duration, s=120, c='green', alpha=0.6, marker='v')
        plt.scatter(self.randomForest_training_mse, self.randomForest_training_duration, s=120, c='orange', alpha=0.6)
        plt.scatter(self.randomForest_test_mse, self.randomForest_test_duration, s=120, c='orange', alpha=0.6, marker='v')
        plt.legend(["TensorFlow Neural Network (Training)", "TensorFlow Neural Network (Inference)", "Linear Regression (Training)", "Linear Regression (Inference)", "Decision Tree Regressor (Training)", "Decision Tree Regressor (Inference)", "Random Forest Regressor (Training)", "Random Forest Regressor (Inference)"], loc="upper center")
        plt.savefig('plots/Algorithms_Evaluation.png')
        #plt.show()
        print("Evaluation Plot saved...")
        print("")
