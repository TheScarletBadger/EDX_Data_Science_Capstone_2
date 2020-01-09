rm(list = ls())
library(knitr)
library(tidyverse)
library(caret)
library(gridExtra)

#RMSE calculation function
rmse = function(yhat,y){
  e = y-yhat
  se = e^2
  mse = mean(se)
  sqrt(mse)}

#read data from file to dataframe
fname = 'C:\\Users\\barry\\Desktop\\Abalone dataset\\abalone.csv'
alldata = data.frame(read.csv(file=fname, header=FALSE, sep=","))

#rename columns as described in documentation
alldata = alldata %>% rename(sex = V1, length = V2, diameter = V3, height = V4,
                             whole_weight = V5, shucked_weight = V6, viscera_weight = V7,
                             shell_weight = V8, rings = V9)
#add new binary variable to denote adult or infant
alldata = alldata %>% mutate(adult = !str_detect(sex,'I'))
alldata[,2:8] = alldata[,2:8]*200

#draw off validation set (20%)
set.seed(1988,sample.kind = 'Rounding')
inds = createDataPartition(alldata$rings, p = 0.2, times = 1, list = FALSE)
trainset = alldata[-inds,]
testset = alldata[inds,]
rm(alldata)
invisible(gc())

#create infant training / validation sets
infant_train = filter(trainset,!adult)
infant_test = filter(testset,!adult)

# Naive benchmark - infant abalone
mu = mean(infant_train$rings)
y = infant_train$rings
yhat = (y*0)+mu
rmse(y,yhat)

#Infant abalone model training / tuning
KNN_tunegrid = data.frame(k = seq(1,101,2))
LM_terminal = train(rings~length+diameter+height+whole_weight+shucked_weight+
                      viscera_weight+shell_weight, method = 'lm',data=infant_train)
GLM_terminal = train(rings~length+diameter+height+whole_weight+shucked_weight+
                       viscera_weight+shell_weight,method = 'glm',data=infant_train)
KNN_terminal = train(rings~length+diameter+height+whole_weight+shucked_weight+
                       viscera_weight+shell_weight,method = 'knn',data=infant_train,
                     tuneGrid = KNN_tunegrid)
LM_nonterminal = train(rings~length+diameter+height+whole_weight,
           method = 'lm',data=infant_train)
GLM_nonterminal = train(rings~length+diameter+height+whole_weight,
           method = 'glm',data=infant_train)
KNN_nonterminal = train(rings~length+diameter+height+whole_weight,
           method = 'knn',data=infant_train,tuneGrid = KNN_tunegrid)

#Infant abalone model testing
lm_t_perf = rmse(predict(LM_terminal,infant_test),t(infant_test$rings))
lm_n_perf = rmse(predict(LM_nonterminal,infant_test),t(infant_test$rings))
glm_t_perf = rmse(predict(GLM_terminal,infant_test),t(infant_test$rings))
glm_n_perf = rmse(predict(GLM_nonterminal,infant_test),t(infant_test$rings))
knn_t_perf = rmse(predict(KNN_terminal,infant_test),t(infant_test$rings))
knn_n_perf = rmse(predict(KNN_nonterminal,infant_test),t(infant_test$rings))

##Infant abalone performance summary table
terminal_pred = c(min(LM_terminal$results["RMSE"]),min(GLM_terminal$results["RMSE"]),
                  min(KNN_terminal$results["RMSE"]))
nonterminal_pred = c(min(LM_nonterminal$results["RMSE"]),min(GLM_nonterminal$results["RMSE"]),
                  min(KNN_nonterminal$results["RMSE"]))
terminal_perf = c(lm_t_perf,glm_t_perf,knn_t_perf)
nonterminal_perf = c(lm_n_perf,glm_n_perf,knn_n_perf)
modelnames = c('Linear model','General linear model','K-nearest neighbours')
perfreport = data.frame(Method = modelnames,terminal_predicted = terminal_pred,terminal_actual = terminal_perf,
                        nonterminal_predicted = nonterminal_pred, nonterminal_actual = nonterminal_perf,
                      loss = nonterminal_perf - terminal_perf)
kable(perfreport, align = c('c','c','c','c','c','c'))

#create adult training / validation sets
adult_train = filter(trainset,adult)
adult_test = filter(testset,adult)

### Naive benchmark - adult abalone
mu = mean(adult_train$rings)
y = adult_train$rings
yhat = (y*0)+mu
rmse(y,yhat)

#Adult abalone Model training / tuning
KNN_tunegrid = data.frame(k = seq(1,101,2))
LM_terminal = train(rings~length+diameter+height+whole_weight+shucked_weight+
                      viscera_weight+shell_weight, method = 'lm',data=adult_train)
GLM_terminal = train(rings~length+diameter+height+whole_weight+shucked_weight+
                       viscera_weight+shell_weight,method = 'glm',data=adult_train)
KNN_terminal = train(rings~length+diameter+height+whole_weight+shucked_weight+
                       viscera_weight+shell_weight,method = 'knn',data=adult_train,
                     tuneGrid = KNN_tunegrid)
LM_nonterminal = train(rings~length+diameter+height+whole_weight,
                       method = 'lm',data=adult_train)
GLM_nonterminal = train(rings~length+diameter+height+whole_weight,
                        method = 'glm',data=adult_train)
KNN_nonterminal = train(rings~length+diameter+height+whole_weight,
                        method = 'knn',data=adult_train,tuneGrid = KNN_tunegrid)

#Adult abalone model testing
lm_t_perf = rmse(predict(LM_terminal,adult_test),t(adult_test$rings))
lm_n_perf = rmse(predict(LM_nonterminal,adult_test),t(adult_test$rings))
glm_t_perf = rmse(predict(GLM_terminal,adult_test),t(adult_test$rings))
glm_n_perf = rmse(predict(GLM_nonterminal,adult_test),t(adult_test$rings))
knn_t_perf = rmse(predict(KNN_terminal,adult_test),t(adult_test$rings))
knn_n_perf = rmse(predict(KNN_nonterminal,adult_test),t(adult_test$rings))

#Adult abalone performance summary table
terminal_pred = c(min(LM_terminal$results["RMSE"]),min(GLM_terminal$results["RMSE"]),
                  min(KNN_terminal$results["RMSE"]))
nonterminal_pred = c(min(LM_nonterminal$results["RMSE"]),min(GLM_nonterminal$results["RMSE"]),
                     min(KNN_nonterminal$results["RMSE"]))
terminal_perf = c(lm_t_perf,glm_t_perf,knn_t_perf)
nonterminal_perf = c(lm_n_perf,glm_n_perf,knn_n_perf)
modelnames = c('Linear model','General linear model','K-nearest neighbours')
perfreport = data.frame(Method = modelnames,terminal_predicted = terminal_pred,terminal_actual = terminal_perf,
                        nonterminal_predicted = nonterminal_pred, nonterminal_actual = nonterminal_perf,
                   loss = nonterminal_perf - terminal_perf)
kable(perfreport, align = c('c','c','c','c','c','c'))


