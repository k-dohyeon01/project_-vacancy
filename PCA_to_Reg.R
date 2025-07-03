rm(list=ls())
install.packages("car")
install.packages("Metrics")
library(tidyverse)
library(caret)
library(car)
library(Metrics)

setwd(choose.dir())
data_train <- read.csv('R분석_train.csv', header = TRUE, na.strings = ',') 
data_test <- read.csv('R분석_test.csv', header = TRUE, na.strings = ',')   

data_train$y <- data_train$y2 / data_train$y1
data_test$y <- data_test$y2 / data_test$y1

data_train_n <- data_train[, c(-1, -2, -3)]
data_test_n <- data_test[, c(-1, -2, -3)]

pca_result <- prcomp(data_train_n[, -ncol(data_train_n)], center = TRUE, scale. = TRUE)
summary(pca_result)

var_explained <- summary(pca_result)$importance["Cumulative Proportion", ]
num_components <- min(which(var_explained >= 0.8))
cat("선택된 주성분 개수:", num_components, "\n")

pca_data <- as.data.frame(pca_result$x[, 1:num_components])
pca_data$y <- data_train_n$y

pca_lm <- lm(y ~ ., data = pca_data)
summary(pca_lm)
vif(pca_lm)

pca_lm_refined <- lm(y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6, data = pca_data)
summary(pca_lm_refined)

train_mean <- colMeans(data_train_n[, -ncol(data_train_n)], na.rm = TRUE)
train_sd <- apply(data_train_n[, -ncol(data_train_n)], 2, function(x) sd(x, na.rm = TRUE))

data_test_scaled <- scale(data_test_n[, -ncol(data_test_n)], center = train_mean, scale = train_sd)
pca_test <- as.matrix(data_test_scaled) %*% pca_result$rotation
pca_test_df <- as.data.frame(pca_test[, 1:num_components])

y_pred_test <- predict(pca_lm_refined, newdata = pca_test_df)
pca_test_mse <- mse(data_test_n$y, y_pred_test)

summary(pca_result)
summary(pca_lm)
cat("전체 PCA 했을 때 test(MSE):", pca_test_mse, "\n")

pca_result$rotation[, 1:6]
