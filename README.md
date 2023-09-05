# Human Pose Dataset and Tasks

Welcome to the Human Pose project! This project is based on a subset of the well-known Human 3.6M dataset, focusing on human actions such as walking, sitting down, giving directions, and posing. These actions are performed by six subjects, and the dataset has been preprocessed to ensure uniformity in actor skeletons.

For more informations on the Human 3.6M dataset, please refer to the following: https://paperswithcode.com/dataset/human3-6m

##

This project is divided into two **milestones**.

For informations on the **first** one please check *"Report Part 1"*.

For informations on the **second** one please check *"Report Part 2"*.


## Dataset Overview

The dataset is divided into three main parts: training, validation, and testing. Each part contains sequences of actions performed by different subjects. In this project, we will tackle two primary tasks:

![image](https://github.com/Lolshh/EPFL-ML-HumanPoses/assets/81919237/6e77806b-dd0b-41ea-8421-b874e1d6779f)

### 1. Action Classification

In the action classification task, we aim to classify sequences of actions into one of four predefined action classes. We analyze sequences lasting 2 seconds to make these classifications.

### 2. Future Pose Regression

For the future pose regression task, we seek to predict the poses in the next 1 second based on sequences lasting 2 seconds.

## Evaluation Metrics

To measure the success of our project, we will use the following metrics for each task:

### For Classification Task:

**Macro F1 Score**: This metric provides an average of class-wise F1 scores, giving us a comprehensive measure of our model's classification performance.

### For Regression Task:

**Mean-Squared Error (MSE)**: MSE will be used to evaluate the accuracy of our regression outputs.

## Techniques and Tools

Throughout the project, we will employ various techniques and tools, including but not limited to:

- **Ridge/Linear Regression**: We'll explore linear regression techniques to understand the relationships within the dataset.
- **Logistic Regression**: Logistic regression will be used for classification tasks.
- **Cross Validation**: Cross-validation ensures robust model evaluation.
- **Principal Component Analysis (PCA)**: PCA may be applied for dimensionality reduction.
- **k-Nearest Neighbors (kNN)**: kNN is a versatile technique for both classification and regression.
- **PyTorch Neural Network**: We will develop PyTorch-based neural networks for classification tasks.
