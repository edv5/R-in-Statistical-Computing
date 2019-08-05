# Apply lasso and ridge regression in different linear models. Compare their performance.

# Generate predictors, errors, X_train and X_test
p = 50
N = 100
set.seed(1)
X_train = array(rnorm(p*N),c(N,p))
eps_train = rnorm(N)
Ntest = 10^3
X_test = array(rnorm(p*Ntest),c(Ntest,p))
eps_test = rnorm(Ntest)
grid = 10^seq(10,-2,length = 100)

library(glmnet)


# Generate the first Y_train and Y_test with Beta = 2
Y_train = 2 * rowSums(X_train[,c(1,2,3,4,5)]) + eps_train
Y_test = 2 * rowSums(X_test[,c(1,2,3,4,5)]) + eps_test

# Run ridge regression over X_train and Y_train
cv.ridge.mod = cv.glmnet(X_train,Y_train,alpha=0,lambda=grid)

# Figure out the best lambda and find out the test error
ridge.bestlam = cv.ridge.mod$lambda.min
ridge.pred = predict(cv.ridge.mod,s=ridge.bestlam,newx=X_test)
mean((ridge.pred - Y_test)^2)
# Test error for ridge regression is 2.312

# Run lasso regression over X_train and Y_train
cv.lasso.mod = cv.glmnet(X_train,Y_train,alpha=1,lambda=grid)

# Then we need to figure out the best lambda and find out the test error
lasso.bestlam = cv.lasso.mod$lambda.min
lasso.pred = predict(cv.lasso.mod,s=lasso.bestlam,newx=X_test)
mean((lasso.pred - Y_test)^2)
# Test error for lasso regression is 1.059


# Generate the second Y_train and Y_test with Beta = 0.5
Y_train_2 = 0.5 * rowSums(X_train) + eps_train
Y_test_2 = 0.5 * rowSums(X_test) + eps_test

# Run ridge regression over X_train2 and Y_train2
cv.ridge.mod_2 = cv.glmnet(X_train,Y_train_2,alpha=0,lambda=grid)

# Figure out the best lambda and find out the test error
ridge.bestlam_2 = cv.ridge.mod_2$lambda.min
ridge.pred_2 = predict(cv.ridge.mod_2,s=ridge.bestlam_2,newx=X_test)
mean((ridge.pred_2 - Y_test_2)^2)
# Test error for ridge regression is 2.327

# Run lasso regression over X_train2 and Y_train2
cv.lasso.mod_2 = cv.glmnet(X_train,Y_train_2,alpha=1,lambda=grid)

# Figure out the best lambda and find out the test error
lasso.bestlam_2 = cv.lasso.mod_2$lambda.min
lasso.pred_2 = predict(cv.lasso.mod_2,s=lasso.bestlam_2,newx=X_test)
mean((lasso.pred_2 - Y_test_2)^2)
# Test error for lasso regression is 2.319


# Create array to hold all the test errors
ridge_test_error = array(NA, c(50))
lasso_test_error = array(NA, c(50))

# Start a for loop for 50 seeds and train the data by using ridge regression over 50 seeds
# Beta is 2 in this case
for (i in 1:50) {
  p = 50
  N = 100
  set.seed(i)
  X_train = array(rnorm(p*N),c(N,p))
  eps_train = rnorm(N)
  Ntest = 10^3
  X_test = array(rnorm(p*Ntest),c(Ntest,p))
  eps_test = rnorm(Ntest)
  grid = 10^seq(10,-2,length = 100)
  
  # We need to first generate Y_train and Y_test
  Y_train = 2 * rowSums(X_train[,c(1,2,3,4,5)]) + eps_train
  Y_test = 2 * rowSums(X_test[,c(1,2,3,4,5)]) + eps_test
  
  #Then we can try to run ridge regression
  cv.ridge.mod = cv.glmnet(X_train,Y_train,alpha=0,lambda=grid)
  
  # Then we need to figure out the best lambda and find out the test error
  ridge.bestlam = cv.ridge.mod$lambda.min
  ridge.pred = predict(cv.ridge.mod,s=ridge.bestlam,newx=X_test)
  
  # Store the ridge test error into the arrary
  ridge_test_error[i] = mean((ridge.pred - Y_test)^2)
  
  # Next, we can run lasso
  cv.lasso.mod = cv.glmnet(X_train,Y_train,alpha=1,lambda=grid)
  
  # Then we need to figure out the best lambda and find out the test error
  lasso.bestlam = cv.lasso.mod$lambda.min
  lasso.pred = predict(cv.lasso.mod,s=lasso.bestlam,newx=X_test)
  
  # Store the lasso test error into the arrary
  lasso_test_error[i] = mean((lasso.pred - Y_test)^2)
}

# We finally generate the boxplots
boxplot(ridge_test_error, main = 'Ridge Test Error')
boxplot(lasso_test_error, main = 'Lasso Test Error')
# From the boxplots, we can clearly see the test error of using lasso regression is lower than test error of using ridge regression.



# Create array to hold all the test errors
ridge_test_error = array(NA, c(50))
lasso_test_error = array(NA, c(50))

# Start a for loop for 50 seeds and train the data by using ridge regression over 50 seeds
# Beta is 0.5 in this case
for (i in 1:50) {
  p = 50
  N = 100
  set.seed(i)
  X_train = array(rnorm(p*N),c(N,p))
  eps_train = rnorm(N)
  Ntest = 10^3
  X_test = array(rnorm(p*Ntest),c(Ntest,p))
  eps_test = rnorm(Ntest)
  grid = 10^seq(10,-2,length = 100)
  
  # We need to first generate Y_train and Y_test
  Y_train = 0.5 * rowSums(X_train) + eps_train
  Y_test = 0.5 * rowSums(X_test) + eps_test
  
  #Then we can try to run ridge regression
  cv.ridge.mod = cv.glmnet(X_train,Y_train,alpha=0,lambda=grid)
  
  # Then we need to figure out the best lambda and find out the test error
  ridge.bestlam = cv.ridge.mod$lambda.min
  ridge.pred = predict(cv.ridge.mod,s=ridge.bestlam,newx=X_test)
  
  # Store the ridge test error into the arrary
  ridge_test_error[i] = mean((ridge.pred - Y_test)^2)
  
  # Next, we can run lasso
  cv.lasso.mod = cv.glmnet(X_train,Y_train,alpha=1,lambda=grid)
  
  # Then we need to figure out the best lambda and find out the test error
  lasso.bestlam = cv.lasso.mod$lambda.min
  lasso.pred = predict(cv.lasso.mod,s=lasso.bestlam,newx=X_test)
  
  # Store the lasso test error into the arrary
  lasso_test_error[i] = mean((lasso.pred - Y_test)^2)
}

# We finally generate the boxplots
boxplot(ridge_test_error, main = 'Ridge Test Error')
boxplot(lasso_test_error, main = 'Lasso Test Error')
# From the boxplots, we can clearly see the test error of using ridge regression is lower than test error of using lasso regression.