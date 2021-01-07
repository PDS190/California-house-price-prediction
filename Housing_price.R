# libraries
library(dummies)
library(scales)
library(forecast)
library(gplots)
library(ggplot2)
library(treemap)
library(reshape)
library(leaps)
library(caret)
library(FNN)
library(lattice)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gains)
library(neuralnet)

# load the data
housing.main <- read.csv("housing.csv", header = TRUE)
housing.df <- housing.main

# summary and dim of the data
summary(housing.df)
sapply(housing.df, class)
dim(housing.df)

# check number of missing values
data.frame(miss.val=sapply(housing.df, function(x)
  sum(length(which(is.na(x))))))

#Omitting rows with missing values
housing.df <- housing.df[complete.cases(housing.df$total_bedrooms), ]
dim(housing.df)

# bar plot
data.for.plot <- aggregate(housing.df$median_house_value, by = list(housing.df$ocean_proximity), FUN = mean)
names(data.for.plot) <- c("ocean proximity", "Meanvalue")
data.for.plot
options(scipen = 999)
barplot(data.for.plot$Meanvalue * 100,  names.arg = data.for.plot$`ocean proximity`,
        xlab = "ocean proximity", ylab = "% of value")

# histogram
par(mfcol = c(3,3))
hist(housing.df$longitude, xlab = "Longitude", main = "Longitude")
hist(housing.df$latitude, xlab = "Latitude", main = "Latitude")
hist(housing.df$housing_median_age, xlab = "age", main = "Median Age")
hist(housing.df$total_rooms, xlab = "number of rooms", main = "Number of Rooms")
hist(housing.df$total_bedrooms, xlab = "number of bedrooms", main = "Number of Bedrooms")
hist(housing.df$population, xlab = "population", main = "Population")
hist(housing.df$households, xlab = "households", main = "Households")
hist(housing.df$median_income, xlab = "Income", main = "Median Income")
hist(housing.df$median_house_value, xlab = "Value", main = "Median Value")

par(mfcol = c(1, 1))#Reset par

# correlation
round(cor(housing.df[, -10]),2)

# create dummies
newhousing.df<-cbind(housing.df, dummy(housing.df$ocean_proximity, sep="_"))
newhousing.df<-newhousing.df[, -10]
summary(newhousing.df)

# Partition data
set.seed(1)
# randomly sample 60% of the row IDs for training; the remaining 40% serve as validation
train.rows <- sample(rownames(newhousing.df), dim(newhousing.df)[1]*0.6)
# collect all the columns with training row ID into training set:
train.data <- newhousing.df[train.rows, ]
# assign row IDs that are not already in the training set, into validation
valid.rows <- setdiff(rownames(newhousing.df), train.rows)
valid.data <- newhousing.df[valid.rows, ]

# dimesion of training and validation data
dim(train.data)
dim(valid.data)
############################################# mlr ##################################################################
mvr <- lm(median_house_value ~ ., data=train.data[, -14])
options(scipen=999) # avoid scientific notation
summary(mvr)

# prediction on validation data
pred <- predict(mvr, newdata = valid.data[, -14])
valid.res <- data.frame(valid.data$median_house_value, pred, residuals =
                          valid.data$median_house_value - pred)
head(valid.res)
hist(valid.res$residuals)

# Error metrics
accuracy(pred, valid.data$median_house_value)

#################################################### KNN ######################################################################
# Normalize the data
# initialize normalized training, validation data, assign (temporarily) data frames to originals
train.norm.df <- train.data
valid.norm.df <- valid.data

norm.values <- preProcess(train.data[, -9], method=c("center", "scale"))
train.norm.df[, -9] <- predict(norm.values, train.data[, -9])
valid.norm.df[, -9] <- predict(norm.values, valid.data[, -9])
summary(train.norm.df)
summary(valid.norm.df)

# initialize a data frame with two columns: k, and RMSE.
accuracy.df <- data.frame(k = seq(1, 20, 1), RMSE = rep(0, 20))

# compute knn for different k on validation.
for(i in 1:20) {
  knn.pred <- knn.reg(train = train.norm.df[, -9], test = valid.norm.df[, -9],
                       train.norm.df[, 9], k = i)
  accuracy.df[i, 2] <- sqrt(mean((valid.data[,9] - knn.pred$pred) ^ 2))
}
accuracy.df

# with best k=11 value
knn11.pred <- knn.reg(train = train.norm.df[, -9], test = valid.norm.df[, -9],
                      train.norm.df[, 9], k = 11)
summary(knn11.pred)
class(knn11.pred)

# actual and prediction
valid.res.knn <- data.frame(actual = valid.norm.df$median_house_value, prediction = knn11.pred$pred)
head(valid.res.knn)

# Error metrics
accuracy(valid.res.knn$prediction,valid.norm.df$median_house_value)

####################################################CART########################################################################
# default tree
default.tree <- rpart(median_house_value ~ ., data = newhousing.df)
summary(default.tree)
options(scipen = 999)
prp(default.tree, type = 1, extra = 1, split.font = 2, varlen = -10)

# plot tree from training data
train.tree <- rpart(median_house_value ~ ., data = train.data)
prp(train.tree, type = 1, extra = 1, split.font = 1, varlen = -10)

# validation data
pred.valid <- predict(train.tree,valid.data)
valid.res_cart <- data.frame(valid.data$median_house_value, pred.valid, residuals =
                          valid.data$median_house_value - pred.valid)
head(valid.res_cart)

# Error metrics
accuracy(pred.valid, valid.data$median_house_value)

# prune tree
cv.ct <- rpart(median_house_value ~ ., data = train.data,
                   cp = 0.00001, minsplit = 10,xval=5)
printcp(cv.ct)

# prune by lower cp
pruned.ct <- prune(cv.ct, cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])
prp(pruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))

# best prune tree
bestpruned.ct <- prune(cv.ct, cp = 0.0026)
prp(bestpruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col=ifelse(bestpruned.ct$frame$var == "<leaf>", 'gray', 'white'))
pred.valid_prun <- predict(bestpruned.ct,valid.data)
valid.res_cart.prune <- data.frame(valid.data$median_house_value, pred.valid_prun, residuals =
                               valid.data$median_house_value - pred.valid_prun)
head(valid.res_cart.prune)

# Error metrics
accuracy(pred.valid_prun, valid.data$median_house_value)

# random forest
rf <- randomForest(median_house_value ~ ., data = data.frame(train.data), ntree = 500,
                       mtry = 4, nodesize = 5, importance = TRUE)

## variable importance plot
varImpPlot(rf, type = 1)

# prediction on validation data
RF.pred.valid <- predict(rf,data.frame(valid.data))
valid.res_rf <- data.frame(valid.data$median_house_value, RF.pred.valid, residuals =
                                     valid.data$median_house_value - RF.pred.valid)
head(valid.res_rf)

# Error metrics
accuracy(RF.pred.valid, valid.data$median_house_value)
########################################################## NN #################################################################3
# rescale the data
train.norm.data <- train.data
valid.norm.data <- valid.data

norm.values.nn <- preProcess(train.data[, -c(9:14)], method="range")
train.norm.data[, -c(9:14)] <- predict(norm.values.nn, train.data[, -c(9:14)])
valid.norm.data[, -c(9:14)] <- predict(norm.values.nn, valid.data[, -c(9:14)])

nn <- neuralnet(median_house_value ~ ., data = data.frame(train.norm.data), linear.output = F, hidden = c(4,4))
plot(nn, rep="best")

# Prediction on validation data
predictnn <- compute(nn, valid.norm.data[,-9])
summary(predictnn)
head(predictnn$net.result)
nnresults <- data.frame(actual = valid.norm.data$median_house_value, prediction = predictnn$net.result)
head(nnresults)

##Unscaled predicted values ##############
unscaled.pred=(nnresults$prediction * diff(range(train.data$median_house_value))) + min(train.data$median_house_value)
accuracy(unscaled.pred,valid.norm.data$median_house_value)
