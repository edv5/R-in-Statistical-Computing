# Build different models by using classification tree, regression tree, pruning, bagging, random forest, boosting techniques 

# Attach carseats dataset from R Library
library(ISLR)
library(tree)
dim(Carseats)
attach(Carseats)

# Create dummy variables for High Sales. If sales > 8, it will be high.
High = ifelse(Sales >= 8, "Yes", "No")
Carseats = data.frame(Carseats, High)


# Divide our data into train and test set
set.seed(2)

# Number of points are 400, 200 training and 200 testing
train = sample(1:nrow(Carseats), 200)
test = Carseats[-train,]
High.test = High[-train]

# Use tree() function to fit a classification tree to predict dependent variable High
tree.carseats = tree(High~. -Sales, Carseats, subset = train)
summary(tree.carseats)
tree.pred = predict(tree.carseats, test, type = "class")

# Calculate the misclassification error on the test set
mean(tree.pred != High.test)
# Missclassification error is 0.285

# Draw the confusion matrix on the test set
table(tree.pred, High.test)

# Plot the tree
plot(tree.carseats)
text(tree.carseats, pretty=0)



# Apply pruning to see whether the tree can have better performance
cv.carseats = cv.tree(tree.carseats, FUN=prune.misclass)
names(cv.carseats)
cv.carseats


# Plot the error rate as a function of both size and k
plot(cv.carseats$size, cv.carseats$dev, type="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")
# From the plot, we can see prune the tree to 9 nodes can lead to the least error

# Prune the tree to obtain nine-node tree
prune.carseats = prune.misclass(tree.carseats, best=9)
plot(prune.carseats)
text(prune.carseats, pretty=0)

# See how well the pruned tree performed
tree.pred = predict(prune.carseats, test, type='class')
mean(tree.pred != High.test)
# Missclassification error is 0.23

# Dtaw the confusion matrix on the test set
table(tree.pred, High.test)



# Fitting Regression Tree. We use Boston dataset this time
library(MASS)
attach(Boston)
nrow(Boston)
train = sample(nrow(Boston), 0.5 * nrow(Boston))
set.seed(1)

#Fit a regression tree on training sample
tree.boston.reg = tree(formula = medv~ ., data=Boston, subset = train)
summary(tree.boston.reg)

# Calculate the error on the test set
tree.boston.pred = predict(tree.boston.reg, newdata = Boston[-train,])
mean((tree.boston.pred-Boston[-train,]$medv)^2)
# The test error is 47.386

# Plot the tree we just modeled
plot(tree.boston.reg)
text(tree.boston.reg, pretty=0)


# Apply pruning to see whether the tree can have better performance
cv.boston=cv.tree(tree.boston.reg)
plot(cv.boston$size, cv.boston$dev, type='b')
cv.boston
# From the plot, we can see prune the tree to 7 nodes can lead to the least error

# Prune the tree to obtain seven-node tree
prune.boston = prune.tree(tree.boston.reg, best=7)
plot(prune.boston)
text(prune.boston, pretty=0)

y.boston.prune = predict(prune.boston, newdata=Boston[-train,])
plot(y.boston.prune, boston.test)
mean((y.boston.prune - boston.test)^2)
# The test error is 50.833, even worse than the tree without pruning


# Bagging
library(randomForest)
set.seed(1)

# Apply bagging on the data to build the model
bag.boston=randomForest(medv ~., data=Boston, subset=train, mtry=13, importance=TRUE)
bag.boston
plot(bag.boston)

# Calculate the test error on the test set
yhat.bag=predict(bag.boston, newdata=Boston[-train,])
plot(yhat.bag, boston.test)
mean((yhat.bag-boston.test)^2)
# The test error is 29.048


# Apply random forest on the data to build the model
#Random Forest p/3 for regression, sqrt(p) for classification
rf.boston=randomForest(medv ~., data=Boston, subset=train, mtry=5, importance=TRUE)
rf.boston
plot(rf.boston)

# Calculate the test error on the test set
yhat.rf= predict(rf.boston, newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
# The test error is 21.693

# Check the importance of each factor
importance(rf.boston)
varImpPlot(rf.boston)

# Apply boosting on the data to build the model
library(gbm)
set.seed(1)
boost.boston=gbm(medv ~., data=Boston[train,], distribution = "gaussian", n.trees=5000, interaction.depth=4)
summary(boost.boston)

# Calculate the test error on the test set
yhat.boost=predict(boost.boston, newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost-boston.test)^2)
# The test error is 22.855