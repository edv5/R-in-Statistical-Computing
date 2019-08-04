# Estimate the logistic regression test error by applying validation set apporoach and 5-fold cross-validation

# Load the default dataset into our model
library(ISLR)
data(Default)
summary(Default)

# Use validation set approach to estimate the test error of default over income and balance
set.seed(1)
half = dim(Default)[1]/2
train_num = sample(1:(2*half),half)
train_sample = Default[train_num,]
train_logistic_reg = glm(default ~ income + balance, family=binomial, data=train_sample)
validation_sample = Default[-train_num,]
validation = (predict(train_logistic_reg, validation_sample[,c('income','balance')], type='response') > 0.5)
validation_sample_yes = (validation_sample[,c("default")] == "Yes")
validation_correct = mean(validation == validation_sample_yes)
validation_error = 1 - validation_correct
validation_error
# The test error is 0.0286

# Use validation set approach to estimate the test error of default over income, balance and student
set.seed(1)
half = dim(Default)[1]/2
train_num_1 = sample(1:(2*half),half)
train_sample_1 = Default[train_num_1,]
train_logistic_reg_1 = glm(default ~ income + balance + student, family=binomial, data=train_sample_1)
validation_sample_1 = Default[-train_num_1,]
validation_1 = (predict(train_logistic_reg_1, validation_sample_1[,c('income','balance','student')], type='response') > 0.5)
validation_sample_yes_1 = (validation_sample_1[,c("default")] == "Yes")
validation_correct_1 = mean(validation_1 == validation_sample_yes_1)
validation_error_1 = 1 - validation_correct_1
validation_error_1
# The test error is 0.0288

# Use 5-fold cross-validation to estimate the test error of default over income, balance and student
library(boot)
set.seed(i)
train_logistic_cv = glm(default ~ income + balance + student, family=binomial, data=Default)
cv.error =cv.glm(Default,train_logistic_cv,K=5)$delta[1]
cv.error
# The test error is 0.0214


