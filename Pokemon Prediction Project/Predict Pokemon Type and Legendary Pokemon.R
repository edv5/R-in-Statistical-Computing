# The code is combined from different codes from each of our team member. 
# There might be some version compatible issues when running the code.
# Some pieces of code can be run successfully in one computer but failed in other computer.

library(MASS)
library(leaps)
library(bestglm)
library(lattice)
library(ggplot2)
library(Matrix)
library(foreach)
library(dummies)
library(data.table)
library(glmnet)
library(readr)
library(ISLR)
library(klaR)
library(caret)
library(nnet)
library(e1071)
library(randomForest)



# 3 Predicting Pokemon Type
# In this case, we use read_csv instead of read.csv
Pokemon <- read_csv("Pokemon.csv")
Pokemon$'Type 2'[is.na(Pokemon$'Type 2')]="null"

# Split the Pokemon dataset into 75% training and 25% testing rondomly
# set.seed(1) to make the result reproductible
set.seed(1)
train_size = floor(0.75 * nrow(Pokemon))
train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
Type1 = as.factor(as.numeric(as.factor(Pokemon$'Type 1')))
Type2 = as.factor(as.numeric(as.factor(Pokemon$'Type 2')))
Pokemon_new = data.frame(Pokemon,Type1,Type2)
train = Pokemon_new[train_ind, ]
test = Pokemon_new[-train_ind, ]

# 3.1 LDA
## type 1
set.seed(1)
lda1_train = subset(train,select=c(6:14))
lda1_test = subset(test,select=c(6:14))
lda_fits_type1 = lda(Type1~.,data=lda1_train,family=binomial)
predict_lda_type1 = predict(lda_fits_type1,lda1_test)
TE_lda_type1 = mean(predict_lda_type1$class != lda1_test$Type1)
TE_lda_type1
Bestvariable_lda_type1 = stepclass(lda1_train[,1:8],lda1_train[,9],method="lda",improvement = 0.001)

## type 2
set.seed(1)
lda2_train = subset(train,select=c(6:15))
lda2_test = subset(test,select=c(6:15))
lda_fits_type2 = lda(Type2~.,data=lda2_train,family=binomial)
predict_lda_type2 = predict(lda_fits_type2,lda2_test)
TE_lda_type2 = mean(predict_lda_type2$class != lda2_test$Type2)
TE_lda_type2
Generation = as.factor(Pokemon$Generation)
Pokemon_lda_type2 = data.frame(Pokemon[,-c(1:5,12)],Generation,Type1,Type2)
Bestvariable_lda_type2 = stepclass(Pokemon_lda_type2[,1:9],Pokemon_lda_type2[,10],method="lda",improvement = 0.001)

# 3.2 QDA
##type 1
set.seed(1)
Pokemon_qda_type1 = data.frame(Pokemon[,-c(1:5,12)],Generation,Type1)
Bestvariable_qda_type1 = stepclass(Pokemon_qda_type1[,1:8],Pokemon_qda_type1[,9],method="qda",improvement = 0.001)


## type 2
set.seed(1)
Pokemon_qda_type2 = data.frame(Pokemon[,-c(1:5,12)],Generation,Type1,Type2)
Bestvariable_qda_type2 = stepclass(Pokemon_qda_type2[,1:9],Pokemon_qda_type2[,10],method="qda",improvement = 0.001)


# 3.3 Multinomial Regression
Pokemon_MR = data.frame(Pokemon[,1:2],Type1,Type2,Pokemon[,5:13])
##type 1
MR_fits_type1 = multinom(Type1~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation, data = Pokemon_MR)
MR_fits_type1
summary(MR_fits_type1)
Bestvariable_MR_type1 = varImp(MR_fits_type1)
Bestvariable_MR_type1$Variables = row.names(Bestvariable_MR_type1)
print(Bestvariable_MR_type1)
funcact = trainControl(method = "cv", number = 5)
CV_MR_Type1=train(Type1~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data=Pokemon_MR,method="multinom",trControl=funcact)
summary(CV_MR_Type1)

##type 2
MR_fits_type2 = multinom(Type2~Type1+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation, data = Pokemon_MR)
MR_fits_type2
summary(MR_fits_type2)
Bestvariable_MR_type2 = varImp(MR_fits_type2)
Bestvariable_MR_type2$Variables = row.names(Bestvariable_MR_type2)
print(Bestvariable_MR_type2)
CV_MR_Type2=train(Type2~Type1+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data=Pokemon_MR,method="multinom",trcontrol=funcact)
print(CV_MR_Type2)


# 3.4 Tree-based Methods

# 1) Setting up and Data inspection
Pokemon = read.csv("Pokemon.csv")
#attach(Pokemon)
summary(Pokemon)
112/800 # Prediction Baseline
unique(Pokemon$Type.1) # 18 unique value in Type 1
unique(Pokemon$Type.2) # 19 unique value in Type 1 (empty count as 1 unique value)
is.na(Pokemon$Type.2) # Empty count as 1 unique value, no need to encode it
#cor(Pokemon)


## Count how many [Type1,Type2] combination
Type.combine <- rep("type", nrow(Pokemon))
for (i in 1:nrow(Pokemon)) {
  if(Pokemon[i,]$Type.2==""){
    Type.combine[i] = c(Pokemon[i,]$Type.1)[1]*100
  }
  else{
    list = sort(c(Pokemon[i,]$Type.1,Pokemon[i,]$Type.2))
    Type.combine[i] = list[1]*100+list[2]
  }
}
ordered(Type.combine)


# 2) Trial run without feature engineering
# Decision Tree without feature engineering
library(tree)
tree.pokemon=tree(Type.1~Total+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,Pokemon)
summary(tree.pokemon)
plot(tree.pokemon)
text(tree.pokemon,pretty=0)
tree.pokemon

# Cross Validation, output: misclassification rate
test_error = rep(0,10)
for (i in 1:10) {
  train_size = floor(0.75 * nrow(Pokemon))
  train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
  Pokemon.train = Pokemon[train_ind, ]
  Pokemon.test = Pokemon[-train_ind, ]
  tree.pokemon=tree(Type.1~Total+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data = Pokemon.train)
  tree.pred=predict(tree.pokemon,Pokemon.test,type="class")
  result <- table(tree.pred,Pokemon.test$Type.1)
  test_error[i] = 1-sum(diag(result))/sum(result)
}
mean(test_error)

# Random Forests without feature engineering
library(randomForest)
bag.pokemon=randomForest(Type.1~Total+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data=Pokemon.train,ntree=500,importance=TRUE,proximities = TRUE)
bag.pokemon
yhat.bag = predict(bag.pokemon,newdata=Pokemon.test)
plot(yhat.bag, Pokemon.test$Type.1)
mean((yhat.bag!=Pokemon.test$Type.1)^2)

# Cross Validation, output: misclassification rate
test_error = rep(0,10)
for (i in 1:10) {
  train_size = floor(0.75 * nrow(Pokemon))
  train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
  Pokemon.train = Pokemon[train_ind, ]
  Pokemon.test = Pokemon[-train_ind, ]
  bag.pokemon=randomForest(Type.1~Total+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data=Pokemon.train,ntree=500,importance=TRUE,proximities = TRUE)
  yhat.bag = predict(bag.pokemon,newdata=Pokemon.test)
  test_error[i] = mean((yhat.bag!=Pokemon.test$Type.1)^2)
}
mean(test_error)


# 3.4.1 Feature Engineering
Pokemon$AtkP = Pokemon$Attack / Pokemon$Total
Pokemon$DefP = Pokemon$Defense / Pokemon$Total
Pokemon$SpAtkP = Pokemon$Sp..Atk / Pokemon$Total
Pokemon$SpDefP = Pokemon$Sp..Def / Pokemon$Total
Pokemon$SpeP = Pokemon$Speed / Pokemon$Total
Pokemon$HpP = Pokemon$HP / Pokemon$Total

Pokemon$AtkToSpatk = Pokemon$Attack / Pokemon$Sp..Atk
Pokemon$DefToSpdef = Pokemon$Defense / Pokemon$Sp..Def

set.seed(2)
train_size = floor(0.75 * nrow(Pokemon))
train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
Pokemon.train = Pokemon[train_ind, ]
Pokemon.test = Pokemon[-train_ind, ]

# 2) Run wit feature engineering
# Decision Tree with feature engineering
library(tree)
tree.pokemon=tree(Type.1~Total+AtkP+DefP+SpAtkP+SpDefP+SpeP+HpP+AtkToSpatk+DefToSpdef+Generation+Legendary,Pokemon)
summary(tree.pokemon)
plot(tree.pokemon)
text(tree.pokemon,pretty=0)
tree.pokemon

# Cross Validation, output: misclassification rate
test_error = rep(0,10)
for (i in 1:10) {
  train_size = floor(0.75 * nrow(Pokemon))
  train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
  Pokemon.train = Pokemon[train_ind, ]
  Pokemon.test = Pokemon[-train_ind, ]
  tree.pokemon=tree(Type.1~Total+AtkP+DefP+SpAtkP+SpDefP+SpeP+HpP+AtkToSpatk+DefToSpdef+Generation+Legendary,data = Pokemon.train)
  tree.pred=predict(tree.pokemon,Pokemon.test,type="class")
  result <- table(tree.pred,Pokemon.test$Type.1)
  test_error[i] = 1-sum(diag(result))/sum(result)
}
mean(test_error)

# Random Forests with feature engineering
library(randomForest)
bag.pokemon=randomForest(Type.1~Total+AtkP+DefP+SpAtkP+SpDefP+SpeP+HpP+AtkToSpatk+DefToSpdef+Generation+Legendary,data=Pokemon.train,ntree=500,importance=TRUE,proximities = TRUE)
bag.pokemon
yhat.bag = predict(bag.pokemon,newdata=Pokemon.test)
plot(yhat.bag, Pokemon.test$Type.1)
mean((yhat.bag!=Pokemon.test$Type.1)^2)

# Cross Validation, output: misclassification rate
test_error = rep(0,10)
for (i in 1:10) {
  train_size = floor(0.75 * nrow(Pokemon))
  train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
  Pokemon.train = Pokemon[train_ind, ]
  Pokemon.test = Pokemon[-train_ind, ]
  bag.pokemon=randomForest(Type.1~Total+AtkP+DefP+SpAtkP+SpDefP+SpeP+HpP+AtkToSpatk+DefToSpdef+Generation+Legendary,data=Pokemon.train,ntree=500,importance=TRUE,proximities = TRUE)
  yhat.bag = predict(bag.pokemon,newdata=Pokemon.test)
  test_error[i] = mean((yhat.bag!=Pokemon.test$Type.1)^2)
}
mean(test_error)



# Model Selection
# Final Models
tree.pokemon=tree(Type.1~Total+AtkP+DefP+SpAtkP+SpDefP+SpeP+HpP+AtkToSpatk+DefToSpdef+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data = Pokemon.train)
tree.pred=predict(tree.pokemon,Pokemon.test,type="class")
summary(tree.pokemon)
result <- table(tree.pred,Pokemon.test$Type.1)
1-sum(diag(result))/sum(result)
plot(tree.pokemon)
text(tree.pokemon,pretty=0)

# Variable Implot
library(randomForest)
set.seed(1)
rf.PokemonType_full = randomForest(Type.1~Total+AtkP+DefP+SpAtkP+SpDefP+SpeP+HpP+AtkToSpatk+DefToSpdef+HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary,data = Pokemon.train, mtry=3, importance = TRUE)
yhat.rfType_full = predict(rf.PokemonType_full, newdata = Pokemon.test)
table(yhat.rfType_full, Pokemon.test$Type.1)
#(172+12)/200 = 0.92
#so random forest here has not improved for the 9 predictor model
varImpPlot(rf.PokemonType_full)




# 4.1 Logistic Regression to Predict Lengerdary Pokemon
# Split the Pokemon dataset into 75% training and 25% testing rondomly
# set.seed(1) to make the result reproductible
Pokemon = read.csv("Pokemon.csv")
summary(Pokemon)
set.seed(1)
train_size = floor(0.75 * nrow(Pokemon))
train_ind = sample(seq_len(nrow(Pokemon)), size = train_size)
train = Pokemon[train_ind, ]
test = Pokemon[-train_ind, ]
selection_model1 = subset(Pokemon,select=c(6,7,8,9,10,11,13))

# Try Logistic Regression with only 6 stats
# There might be some issues due to different version of R Studio.
# If misclassification error shows 1, changing FALSE to False and TRUE to True can solve this
Lo_Reg1_train = subset(train,select=c(6,7,8,9,10,11,13))
Lo_Reg1_test = subset(test,select=c(6,7,8,9,10,11,13))
glm.fits1 = glm(Legendary~.,data=Lo_Reg1_train,family=binomial)
fitted.results1 = predict(glm.fits1,newdata=Lo_Reg1_test,type='response')
glm.pred1 = rep(FALSE,200)
glm.pred1[fitted.results1>.5]= TRUE
misClasificError1 = mean(glm.pred1 != test$Legendary)
table(glm.pred1 , test$Legendary)
print(paste('Misclassification Error is ', misClasificError1))

# Since we only have 6 features in this case, we can simply use best subset regression to see what variables to include
# We use cross-validation 
bestglm(selection_model1,IC="CV",family=binomial, method="exhaustive")
# Best Subset Selection chooses the model without sp. defense and attack

# Let's see how the model without sp. defense and attack looks like
# There might be some issues due to different version of R Studio.
# If misclassification error shows 1, changing FALSE to False and TRUE to True can solve this
Lo_Reg2_train = subset(train,select=c(6,8,9,11,13))
Lo_Reg2_test = subset(test,select=c(6,8,9,11,13))
glm.fits2 = glm(Legendary~.,data=Lo_Reg2_train,family=binomial)
fitted.results2 = predict(glm.fits2,newdata=Lo_Reg2_test,type='response')
glm.pred2 =rep(FALSE,200)
glm.pred2 [fitted.results2>.5]=TRUE
misClasificError2 = mean(glm.pred2 != Lo_Reg2_test$Legendary)
table(glm.pred2 , Lo_Reg2_test$Legendary)
print(paste('Misclassification Error is ', misClasificError2))

# Right now, we can expand the model a little bit, by including not only 6 stats, but also generation and types
# To achieve this, we can use package dummies to transform these factors into dummy variables
rows = seq.int(800)
type1_dummy = dummy(Pokemon$Type.1)
type2_dummy = dummy(Pokemon$Type.2)
type_dummy = type1_dummy + type2_dummy[,c(2:19)]
generation_dummy = dummy(Pokemon$Generation)
type_dummy = as.data.frame.matrix(type_dummy)
generation_dummy = as.data.frame.matrix(generation_dummy)
type_dummy$ID = rows
generation_dummy$ID = rows
temp1 = merge(type_dummy, generation_dummy, by=c("ID"))
Pokemon_temp1 = Pokemon[,c(6:11)]
Pokemon_temp2 = Pokemon[,c(13,12)]
Pokemon_temp1$ID = rows
Pokemon_temp2$ID = rows
Pokemon_all_variable = merge(Pokemon_temp1, temp1, by=c("ID"))
Pokemon_all_variable = merge(Pokemon_all_variable, Pokemon_temp2, by=c("ID"))
Pokemon_all_variable = Pokemon_all_variable[,c(2:32)] 

# Try the full logistic model
# There might be some issues due to different version of R Studio.
# If misclassification error shows 1, changing FALSE to False and TRUE to True can solve this
train_full = Pokemon_all_variable[train_ind, ]
test_full = Pokemon_all_variable[-train_ind, ]
glm.fits_full = glm(Legendary~.,data=train_full,family=binomial)
fitted.results_full = predict(glm.fits_full,newdata=test_full,type='response')
glm.pred_full =rep(FALSE,200)
glm.pred_full[fitted.results_full>.5]=TRUE
misClasificError_full = mean(glm.pred_full != test_full$Legendary)
table(glm.pred_full, test_full$Legendary)
print(paste('Misclassification Error is ', misClasificError_full))

# Try Lasso
grid = 10^seq(10,-2,length = 100)
full_lasso = cv.glmnet(x=as.matrix(train_full[,c(1:30)]), y=as.factor(train_full[,c(31)]), alpha=1, family="binomial",lambda=grid)
lasso.bestlam = full_lasso$lambda.min
full_lasso_pred = predict(full_lasso,s=lasso.bestlam,newx=as.matrix(test_full[,c(1:30)]), family="binomial")
glm.pred3 =rep(FALSE,200)
glm.pred3 [full_lasso_pred>0]=TRUE
misClasificError3 = mean(glm.pred3 != test_full$Legendary)
table(glm.pred3 , test_full$Legendary)
print(paste('Misclassification Error is ', misClasificError3))

# Let's go through forward stepwise selection on the full model by comparing BIC
forward_glm.fits_full = stepAIC(glm.fits_full, k=log(nrow(train_full)), direction = "forward", trace = FALSE)
forward_glm.fits_full$anova
# Foward selection turns out the full model

# Next, we go through backward stepwise selection on the full model by comparing BIC
backward_glm.fits_full = stepAIC(glm.fits_full, k=log(nrow(train_full)), direction = "backward", trace = FALSE)
backward_glm.fits_full$anova
# It turns out the the model we use backward stepwise selection only includes HP, attack, sp. attack, sp. defense and speed

# Let's see how the model with only HP, attack, sp. attack, sp. defense and speed looks like
train_full2 = subset(train,select=c(6,7,9,10,11,13))
test_full2 = subset(test,select=c(6,7,9,10,11,13))
glm.fits3 = glm(Legendary~.,data=train_full2,family=binomial)
fitted.results3 = predict(glm.fits3,newdata=test_full2,type='response')
glm.pred3 =rep("FALSE",200)
glm.pred3 [fitted.results3>.5]="TRUE"
misClasificError3 = mean(glm.pred3 != test_full2$Legendary)
table(glm.pred3 , test_full2$Legendary)
print(paste('Misclassification Error is ', misClasificError3))






