install.packages("MASS")
install.packages("caret")
install.packages("dplyr")
install.packages("ggplotify")

library(ggplot2)
library(ggplotify)
library(dplyr)
library(caret)
library(class)

# Reading the dataset
d = read.csv("E://Ketaki S//Breast_Cancer_Wisconsin_Data_Set.csv")

# Viewing the data
View(d)

# Checking the dimensions of the dataset
dim(d)

# Displaying the column names
colnames(d)

# Viewing the first few rows of the data
head(d)

# Removing unnecessary columns: ID column (1st) and an extra empty column (33rd)
d = d[,-c(1,33)]

# Checking for missing or null values in the data
sum(is.na(d))

# Getting a summary of the entire dataset
summary(d)

# Checking the distribution of the target class (diagnosis)
b = dim(d[which(d$diagnosis == "0"),])[1]
m = dim(d[which(d$diagnosis == "1"),])[1]

# Plotting the class distribution
barplot(c(b, m), col = c("pink", "blue"), names.arg = c("benign", "malignant"), 
        main = "Total Count of Each Class", ylab = "Counts")
View(d)

# Scaling the data (excluding the target class) to ensure variables are on the same scale
d_sc = scale(d[,-1], center = TRUE, scale = TRUE)

# Adding the target variable back to the scaled data
dd_scale = cbind(d_sc, d$diagnosis)
colnames(dd_scale)[31] = "class"
dd_scale = as.data.frame(dd_scale)

# Getting a summary of the scaled dataset
summary(dd_scale)

# Checking the correlation between features
mean(cor(dd_scale))

# Viewing the first few rows of the scaled dataset
head(dd_scale)

# Checking the dimensions of the scaled dataset
dim(dd_scale)

# Applying PCA due to multicollinearity and high dimensionality, which might affect model performance
library(stats)
install.packages("factoextra")
library(factoextra)

# Performing PCA on the scaled dataset (excluding the target class)
PCA = princomp(dd_scale[,-31])

# Displaying a summary of PCA results
summary(PCA)

# Scree plot to visualize the percentage of variance explained by each principal component
fviz_eig(PCA, addlabels = TRUE)

# Selecting the first 6 principal components that explain about 89% of the variance
pc6 = PCA$scores[,c(1,2,3,4,5,6)]
dd_scale1 = data.frame(pc6, dd_scale$class)
colnames(dd_scale1)[7] = "class"

# Displaying the PCA loadings for the first 6 components
PCA$loadings[,1:6]

# Determining the optimal number of clusters using the "within sum of squares" method
my_pca = data.frame(pc6)
fviz_nbclust(my_pca, FUNcluster = kmeans, method = "wss")

# The optimal number of clusters is found to be 3
Kmeans = kmeans(my_pca, centers = 2)

# Checking the number of observations in each cluster
table(Kmeans$cluster)

# Plotting the clusters
rownames(my_pca) = paste(d$diagnosis, 1:dim(d)[1], sep = "_")
fviz_cluster(list(data = my_pca, cluster = Kmeans$cluster))

# Cross-checking the actual diagnosis with the clusters formed
table(d$diagnosis, Kmeans$cluster)

# Splitting the data into training and testing sets (70% training, 30% testing)
n = dim(dd_scale1)[1]
set.seed(700)
ind = sample(1:n, n * 0.7)
train_data = dd_scale1[ind,]
test_data = dd_scale1[-ind,]

# Verifying the dimensions of the train and test sets
dim(train_data)
head(train_data)

# Fitting a Naive Bayes model on the training data
library(e1071)
naive_model = naiveBayes(class ~ ., data = train_data)

# Predicting on the test data
pre = predict(naive_model, test_data[,-7])
pre

# Evaluating the Naive Bayes model using a confusion matrix
cm = confusionMatrix(as.factor(test_data$class), as.factor(pre))
cm

# Calculating the F1 score for Naive Bayes
f1_score1 = cm$table[1] / (cm$table[1] + 0.5 * (cm$table[3] + cm$table[2]))
f1_score1

# Fitting a KNN model and evaluating for different values of k
k_values = 1:20
f1_scores = numeric(length(k_values))

for (i in k_values) {
  cat("----------------- For k =", i, "-----------------\n")
  knn_model = knn(train_data, test_data, train_data$class, k = i)
  cm = confusionMatrix(knn_model, as.factor(test_data$class))
  f1_score = cm$table[1] / (cm$table[1] + 0.5 * (cm$table[3] + cm$table[2]))
  f1_scores[i] = f1_score
  cat("F1 score is: ", f1_score, "\n")
  cat("\n")
}

# Plotting F1 score vs k for KNN model
plot(k_values, f1_scores, type = "b", pch = 19, col = "blue",
     xlab = "k", ylab = "F1 Score", main = "F1 Score vs. k for KNN Model")
abline(v = 3, col = "red", lty = 2)

# Setting the optimal value of k as 3
knn_model = knn(train_data, test_data, train_data$class, k = 3)
cm = confusionMatrix(knn_model, as.factor(test_data$class))

# Calculating the F1 score for KNN with k=3
f1_score2 = cm$table[1] / (cm$table[1] + 0.5 * (cm$table[3] + cm$table[2]))

# Fitting a logistic regression model
glm_model = glm(class ~ ., family = binomial, data = train_data)
summary(glm_model)

# Predicting on the test data using the logistic regression model
pre = predict(glm_model, test_data[,-7], type = "response")

# Converting predicted probabilities to binary outcomes
pre1 = ifelse(pre > 0.5, 1, 0)
pre1

# Calculating the accuracy of the logistic regression model
accuracy = mean(pre1 == test_data$class)
cat("Accuracy of Logistic Regression Model:", accuracy, "\n")

# Creating a confusion matrix for the logistic regression model
conf_matrix = table(Actual = test_data$class, Predicted = pre1)
conf_matrix

# Calculating precision, recall, and F1 score for the logistic regression model
precision = conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall = conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score3 = 2 * (precision * recall) / (precision + recall)

cat("F1 Score for Logistic Regression:", f1_score3, "\n")


# Comparing F1 scores across models
d = data.frame(Models = c("Naive Bayes", "KNN", "Logistic Regression"),
               f1_score = c(f1_score1, f1_score2, f1_score3))

# Plotting F1 scores for different models
barplot(c(f1_score1, f1_score2, f1_score3),
        names.arg = c("Naive Bayes", "KNN", "Logistic Regression"),
        col = c("pink", "blue", "light green"),
        ylim = c(0, 1), main = "F1 Scores of Different Models")
