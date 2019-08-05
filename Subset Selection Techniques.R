# Apply subset selection techniques to perform feature engineering

# Attach Hitters dataset from ISLR
library(ISLR)
attach(Hitters)

# Discover that there are 59 values are missing
sum(is.na(Hitters))

# Apply best subset selection to identify what predictors we should include 
library(leaps)
regfit.full = regsubsets(Salary~., Hitters, nvmax = 19)
reg.summary = summary(regfit.full)

# Check r-square of each number of variables selected
reg.summary$rsq

# Plot the RSS of each number of variables selected. Find the min RSS and plot the point as red
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS")
min.rss = which.min(reg.summary$rss)
points(min.rss, reg.summary$rss[min.rss], col="red", cex=2, pch=20)
# min.rss = 19

# Plot the adjusted RSQ of each number of variables selected. Find the max adjusted RSQ and plot the point as red
plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSQ")
max.rsq = which.max(reg.summary$adjr2)
points(max.rsq, reg.summary$adjr2[max.rsq], col='red', cex=2, pch=20)
# max.rsq = 11

# Plot the Cp of each number of variables selected. Find the min Cp and plot the point as red
plot(reg.summary$cp, xlab="Number of Variables", ylab="Cp")
min.cp = which.min(reg.summary$cp)
points(min.cp, reg.summary$cp[min.cp], col="red", cex=2, pch=20)
# min.cp = 10

# Plot the BIC of each number of variables selected. Find the min BIC and plot the point as red
plot(reg.summary$bic, xlab="Number of Variables", ylab="BIC")
min.bic = which.min(reg.summary$bic)
points(min.bic, reg.summary$bic[min.bic], col="red", cex=2, pch=20)
# min.bic = 6

# Plot the best model selection by referencing rss, adjusted rsq, Cp and BIC
plot(regfit.full, scale='r2')
plot(regfit.full, scale='adjr2')
plot(regfit.full, scale='Cp')
plot(regfit.full, scale='bic')

# From the plots, we can see that six-variable model has the lowest BIC
# Use coefficient estimates to see the variables and their coefficients
coef(regfit.full, 6)
# The 6 coefficients are AtBat, Hits, Walks, CRBI, DivisionW and Putouts



# Apply forward selection to identify what predictors we should include
regfit.fwd = regsubsets(Salary~., Hitters, nvmax = 19, method = "forward")
fwd.summary = summary(regfit.fwd)

# Plot the BIC of each number of variables selected. Find the min BIC and plot the point as red
plot(fwd.summary$bic, xlab="Number of Variables", ylab="BIC")
fwd.min.bic = which.min(fwd.summary$bic)
points(fwd.min.bic, fwd.summary$bic[fwd.min.bic], col="red", cex=2, pch=20)
# min.bic = 6

# Plot the best model selection by referencing BIC
plot(regfit.fwd, scale='bic')

# From the plots, we can see that six-variable model has the lowest BIC
# Use coefficient estimates to see the variables and their coefficients
coef(regfit.fwd, 6)
# The 6 coefficients are AtBat, Hits, Walks, CRBI, DivisionW and Putouts (same as best subset selection)



# Apply backward selection to identify what predictors we should include
regfit.bwd = regsubsets(Salary~., Hitters, nvmax = 19, method = "backward")
bwd.summary = summary(regfit.bwd)

# Plot the BIC of each number of variables selected. Find the min BIC and plot the point as red
plot(bwd.summary$bic, xlab="Number of Variables", ylab="BIC")
bwd.min.bic = which.min(bwd.summary$bic)
points(bwd.min.bic, bwd.summary$bic[bwd.min.bic], col="red", cex=2, pch=20)
# min.bic = 8

# Plot the best model selection by referencing BIC
plot(regfit.bwd, scale='bic')

# From the plots, we can see that eight-variable model has the lowest BIC
# Use coefficient estimates to see the variables and their coefficients
coef(regfit.bwd, 8)
# The 8 coefficients are AtBat, Hits, Walks, CRuns, CRBI, Cwalks, DivisionW and Putouts