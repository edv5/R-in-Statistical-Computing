# Apply resampling techniques, such as validation set approach, LOOCV and K-fold cross-validation to the dataset to find the best fitting model
# Apply bootstrap to estimate variance
# Attach Carseats dataset from ISLR library
library(ISLR)
attach(Carseats)


# Apply validation Set Approach
# Split the data into half and half
set.seed(1)
train = sample(nrow(Carseats), 0.5 * nrow(Carseats))

# Use validation set approach to see whether we should fit linear, quadratic or cubic regression over income with Sales
lm.fit = lm(Sales~Income, data=Carseats, subset=train)
mean((Sales-predict(lm.fit, Carseats))[-train]^2)
# Error is 8.287

lm.fit2 = lm(Sales~poly(Income,2), data=Carseats, subset=train)
mean((Sales-predict(lm.fit2, Carseats))[-train]^2)
# Error is 8.453

lm.fit3 = lm(Sales~poly(Income,3), data=Carseats, subset=train)
mean((Sales-predict(lm.fit3, Carseats))[-train]^2)
# Error is 8.488
# Therefore, linear is better


# Use LOOCV to see whether we should fit linear, quadratic or cubic regression over income with Sales
library(boot)

#LOOCV from linear to 3 times
cv.error=rep(0,3)

for (i in 1:3) {
  glm.fit=glm(Sales~poly(Income,i), data=Carseats)
  cv.error[i] = cv.glm(Carseats, glm.fit)$delta[1]
}
cv.error
# The errors are 7.848, 7.876, 7.912, respectively.
# Therefore, linear is the best model.


# Use K-fold cross-validation to see whether we should fit linear, quadratic or cubic regression over income with Sales
# Here, we use 10-fold CV
set.seed(1)

cv.error.10=rep(0,3)
for (i in 1:3) {
  glm.fit=glm(Sales~poly(Income,i), data=Carseats)
  cv.error.10[i] = cv.glm(Carseats, glm.fit, K=10)$delta[1]
}
cv.error.10
# The errors are 7.821, 7.919, 7.915, respectively.
# Therefore, linear is the best model.

# Attach portfolio dataset from ISLR on bootstrap
attach(Portfolio)

# Create a function alpha.fn(), taking (X,Y) data.
# The output is an estimate for alpha based on the selecte observations.
alpha.fn = function(data, index) {
  X = data$X[index]
  Y = data$Y[index]
  return ((var(Y) - cov(X,Y)) / (var(X) + var(Y) - 2 * cov(X,Y)))
}


# Apply bootstrap to estimate variance
# Randomly select 100 observations from portfolio with replacemnt
set.seed (1)
alpha.fn(Portfolio, sample(100, 100, replace = T))
# alpha = 0.596

# Implement bootstrap analysis on this command many times
# Produce R=1000 bootstrap estimates for alpha.
boot(Portfolio, alpha.fn, R = 1000)
# alpha_hat = 0.576, bootstrap estimate for standard error is 0.089
