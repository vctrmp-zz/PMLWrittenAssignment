library(parallel)
library(doParallel)
library(caret)
library(RCurl)


# download source data from he web
download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
              destfile = "pml-training.csv")

download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
              destfile = "pml-testing.csv")

# load csvs
pTraining <- read.csv("pml-training.csv")
pTest <- read.csv("pml-testing.csv")

# peek at the data
head(pTraining)

# removing the first 7 columns from both files as these columns does not seem to help training my models
pTraining <- pTraining[,-c(1:7)]
pTest <- pTest[,-c(1:7)]

# remove Near Zero Variance columns from the training set, and remove the same from the test set
nzv <- nearZeroVar(pTraining)
pTraining <- pTraining[, -nzv]
pTest <- pTest[, -nzv]

# remove training N/A columns from the training set, and remove the same from the test setjjjjjj
cc <- complete.cases(t(pTraining))
pTraining <- pTraining[,cc]
pTest <- pTest[,cc]

# set seed for repoductibility, splitting the training set in 60% / 40% (training / validation)
set.seed(501)
partTrain <- createDataPartition(y = pTraining$classe, p = 0.6, list=FALSE)
training <- pTraining[partTrain,]
testing <- pTraining[-partTrain,]


cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
# stopCluster(cl)

set.seed(501)
system.time(gbm <- train(classe ~ ., data=training, method="gbm"))
set.seed(501)
system.time(lda <- train(classe ~ ., data=training, method="lda"))
set.seed(501)
system.time(nb <- train(classe ~ ., data=training, method="nb"))
set.seed(501)
system.time(pls <- train(classe ~ ., data=training, method="pls"))
set.seed(501)
system.time(rf <- train(classe ~ ., data=training, method="rf"))
set.seed(501)
system.time(rpart <- train(classe ~ ., data=training, method="rpart"))
set.seed(501)
system.time(xgb <- train(classe ~ ., data=training, method="xgbTree"))

# Results analysis
results <- resamples( list(GBM=gbm, LDA=lda, NF=nb, PLS=pls, RF=rf, RPart=rpart, 
                           XGBoost=xgb))
summary(results)
bwplot(results)

# Add parameters to RF, my chosen model
control <- trainControl(method="repeatedcv", number=10, repeats=5)
set.seed(501)
rfMod <- train(classe ~ ., data=training, method="rf", trControl=control)
rfModError <- confusionMatrix(predict(rfMod, newdata=testing), testing$classe)

#Submission
pred <- predict(xgb, newdata = pTest)
pred

# Saving Fits 
saveRDS(rf, "rf.rds")
saveRDS(rfMod, "rfMod.rds")
saveRDS(rfError, "rfError.rds")

rf <- readRDS("rf.rds")
rfMod <- readRDS("rfMod.rds")
rfError <- readRDS("rfError.rds")

rf
rfError <- confusionMatrix(predict(rf, newdata=training), training$classe)
rfError

rfModError
1 - 0.992
