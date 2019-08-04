# Application of classification models such as LDA, QDA, logistic regression and KNN

library(ISLR)
attach(Auto)

# Create a binary variable mpg01 that contains a 1 if mpg is above median and a 0 otherwise
mpg01 = rep(0, length(mpg))
mpg01[mpg > median(mpg)] = 1

# Show some visualization over mpg and other factors
Auto_new = data.frame(Auto, mpg01)
pairs(Auto_new)
boxplot(cylinders ~ mpg01, data=Auto_new, main="cylinders vs mpg01")
boxplot(displacement ~ mpg01, data=Auto_new, main="displacement vs mpg01")
boxplot(horsepower ~ mpg01, data=Auto_new, main="horsepower vs mpg01")
boxplot(weight ~ mpg01, data=Auto_new, main="weight vs mpg01")
boxplot(acceleration ~ mpg01, data=Auto_new, main="acceleration vs mpg01")
boxplot(year ~ mpg01, data=Auto_new, main="year vs mpg01")
boxplot(origin ~ mpg01, data=Auto_new, main="origin vs mpg01")

# Split the data into a training and test set
set.seed(1)
half = dim(Auto_new)[1]/2
train_num = sample(1:(2*half),half)
train_sample = Auto_new[train_num,]
validation_sample = Auto_new[-train_num,]

# Apply LDA on the training data use cylinders, displacement, horsepower and weight as independent variables
library(MASS)
lda_auto_new = lda(mpg01 ~ cylinders + displacement + horsepower + weight, data = train_sample)
lda_auto_new

# Obtain the test error
validation = predict(lda_auto_new, validation_sample)
validation_correct = mean(validation$class == validation_sample$mpg01)
validation_error = 1 - validation_correct
validation_error
# Test error of LDA is about 0.102

# Apply QDA on the training data by using cylinders, displacement, horsepower and weight as independent variables
qda_auto_new = qda(mpg01 ~ cylinders + displacement + horsepower + weight, data = train_sample)
qda_auto_new

validation = predict(qda_auto_new, validation_sample)
validation_correct = mean(validation$class == validation_sample$mpg01)
validation_error = 1 - validation_correct
validation_error
# Test error of QDA is about 0.122

# Apply logistic regression on the training data by using cylinders, displacement, horsepower and weight as independent variables
logistic_auto_new = glm(mpg01 ~ cylinders + displacement + horsepower + weight, data = train_sample, family=binomial)
summary(logistic_auto_new)
logistic_validation = predict(logistic_auto_new, validation_sample) > 0.5
mean(logistic_validation != validation_sample$mpg01)
# Test error of logistic regression is about 0.077

# Draw the confusion matrix
predict_result = rep(0, nrow(validation_sample))
predict_result[logistic_validation == TRUE] = 1
predict_result[logistic_validation == FALSE] = 0
table(predict_result, validation_sample$mpg01)


# Draw the ROC curve to further check the model accuracy
library(ROCR)
pred = prediction(predict(logistic_auto_new, validation_sample), validation_sample$mpg01)
auc_perf = performance(pred, measure = "auc")
plot(performance(pred, "tpr", "fpr"))

# Calculate the Gini Index
library(MLmetrics)
2*auc_perf@y.values[[1]]-1
Gini(y_pred = predict(logistic_auto_new, validation_sample), y_true = validation_sample$mpg01)


# Apply KNN on the training data by using cylinders, displacement, horsepower and weight as independent variables
library(class)
features = c("cylinders", "displacement", "horsepower", "weight")
knn_error = rep(0,10)
for (i in 1:10) {
  knn = knn(train_sample[features], validation_sample[features], train_sample$mpg01, k=i)
  knn_error[i] = 1 - mean(knn == validation_sample$mpg01)
}
knn_error
# When k = 6, the min error is 0.092
