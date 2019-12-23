library(foreign)
library(dplyr)
THRdata = read.arff("ThoracicSurgery.arff") #Reading in Thoracic Surgery Data

#Cleaning up dataset by renaming columns
THRdata = THRdata %>% rename(Diagnosis=DGN, FVC=PRE4, FEV=PRE5, Performance=PRE6, Pain=PRE7, Haemoptysis=PRE8, 
                       Dyspnea=PRE9, Cough=PRE10, Weakness=PRE11, TumorSize=PRE14, DM=PRE17, MI6mo=PRE19,
                       PAD=PRE25, Smoking=PRE30, Asthma=PRE32, Age=AGE, Death1Y=Risk1Yr)

#Making data matrix out of dataset for easier comprehension for modeling
#THRmatrix = data.matrix(THRdata)

#Creating training and test datasets
set.seed(1127)
n <- 0.70*nrow(THRdata) 
smp <- sample(nrow(THRdata), size = n, replace = FALSE) 
#trainset <- THRmatrix[trainsmp,]
trainfac <- THRdata[smp,]
#testset <- THRmatrix[-trainsmp,]
testfac <- THRdata[-smp,]

#randomForest feature selection
library(randomForest)
THRrf = randomForest(Death1Y ~ ., data = trainfac, importance = TRUE, oob.times = 15, confusion = TRUE)
importance(THRrf)
varImpPlot(THRrf)

#RFE feature selection
library(lattice)
library(caret)
control = rfeControl(functions = caretFuncs, number = 2)
THRrfe = rfe(trainfac[,1:16], trainfac[,17], sizes = c(3, 5, 12), rfeControl = control)
THRrfe

#Training model using recursive partitioning model with binomial prediction and integrated cross validation
#In the train control statement, we also use a 'ROSE' sampling method which employs a hybrid of over- and
#under- sampling. We use a sampling method because of the imbalanced data in the dataset which overaccounts for
#one class of classification.
set.seed(1127)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10,
                              sampling = "rose",
                              repeats = 10, 
                              verboseIter = FALSE)
rpartTHR1 = train(Death1Y ~ FVC + FEV + Diagnosis + TumorSize + Age, 
                 data = trainfac,
                 trControl = train.control,
                 preProcess = c("scale", "center"),
                 method = "rpart")

#Using our model, we predict our binary classifier on the test data. We create a confusion matrix to evaluate
#results, showing a ~83% accuracy and 99% sensitivity.
THRpredict <- data.frame(actual = testfac$Death1Y,
                         predict(rpartTHR1, newdata = testfac, type = "prob"))
THRpredict$predict <- ifelse(THRpredict$T > 0.5, "F", "T")
cm_rose <- confusionMatrix(as.factor(THRpredict$predict), testfac$Death1Y)
cm_rose

#Using the ROSE package, we create a ROC curve. The resulting AUC is poor even after sampling the imbalanced data
#so further data cleanup may be necessary.
#We obtain a AUC of 58.6%
roc.curve(THRpredict$predict, testfac$Death1Y)

################################## Other attempted code.
# rpartTHR = train(Death1Y ~ FVC + FEV + Diagnosis + TumorSize + Age, 
#                 data = trainfac,
#                 trControl = train.control,
#                 method = "rpart")
# print(rpartTHR)
# summary(rpartTHR)
# confusionMatrix(rpartTHR)

# logisregTHR = glm(Death1Y ~ FVC + FEV + Diagnosis + TumorSize + Age, data = trainfac, family = binomial)
# summary(logisregTHR)
# 
# set.seed(1127)
# THRpredict = predict(logisregTHR, newdata = testfac, type = "response")
# numpred <- ifelse(THRpredict > 0.5, "T", "F") 
# THRpredict <- factor(numpred, levels = c("F", "T"))
# confusionMatrix(THRpredict, testfac$Death1Y)

# library(prediction)
# library(performance)
# set.seed(1127)
# pr <- prediction(as.numeric(THRpredict), testfac$Death1Y)
# perf <- performance(pr, "tpr", "fpr")
# plot(perf)


# THRauc = roc(THRpredict)
# plot(THRauc)
# plot.roc(THRdata$Death1Y, as.numeric(THRpred))

         