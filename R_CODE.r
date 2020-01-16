rm(list=ls())   #clean the RAM
setwd("C:/Users/user/Documents")   #set working directory
getwd()             #get current working directory
df=read.csv("C:/Users/user/Downloads/train.csv",header=T)    #loading train dataset
df2=read.csv("C:/Users/user/Downloads/test.csv",header=T)   #loading test dataset
dim(df)          #getting no of observations and variables in train dataset
dim(df2)         #getting no of observations and variables in test dataset
str(df)         #getting structure variables names,their type and data of train dataset
str(df2)         #getting structure variables names,their type and data of test dataset
colnames(df)     #column names of train dataset
colnames(df2)     #column names of test dataset
class(df$target)    #type of variable target in train dataset
class(df2$var_0)    #type of variable var_0 in test dataset
miss_val=data.frame(apply(df,2,function(target){
  sum(is.na(target))
}))
miss_val
df[20,3]     #get value at 20th observation for 3rd variable in train dataset        # 4.409
df[20,3]=NA      #set value NA at 20th observation for 3rd variable in train dataset
df$var_0[is.na(df$var_0)]=mean(df$var_0,na.rm=T)  #Mean method           
df[20,3]                                 # 10.67995
df$var_0[is.na(df$var_0)]=median(df$var_0,na.rm=T)        #Median method
df[20,3]                                   # 10.5248
require(DMwR)                       #library to impute KNN Imputation for Missing value Analysis
df=knnImputation(df, k=2)      #Knn Imputation
sum(is.na(df))
df[20,3]                                  # 11.0208
df <- data.frame(df[,-1], row.names = df[,1])       #make the first column as index for the train dataset
df
df2 <- data.frame(df2[,-1], row.names = df2[,1]) #make the first column as index for the test dataset
df$target=as.numeric(df$target)                    #changing target to numeric type
require(corrplot)                                            #library to plot correlation graph
M<-cor(df)                                                     
head(round(M,2))
corrplot(M1, method="circle")                       #plot correlation graph indicating circles
corrplot(M1, method="color")                       #plot correlation graph indicating colours
corrplot(M1, method="number")                       #plot correlation graph indicating numbers
require(tibble)                                   #library to add column at given location in dataset
df2=add_column(df2,target=df$target,.before="var_0")      #adding target column in test   dataset before var_0
df2$target=as.factor(df2$target)     #converting target variable to factor variable as it is categorical variable
df$target=as.factor(df$target)    #converting target variable to factor variable as it is categorical variable
require(glmm)                #library to perform Logistic Regression
logit_model=glm(target~.,data=df,family="binomial")     
summary(logit_model)
pred1=predict(logit_model,newdata=df2,type="response")   #predicting values for test data
pred1=ifelse(pred1>=0.05,1,0)                        #if probability is > 0.05 then 1 else 0
pred1
conf=table(df$target,pred1)         #build confusion matrix
conf      
83/(83+55)                                   #determine False Negative Rate        #0.6014493
(55+7)/(55+83+7+5)                    #determine accuracy                          #0.4133333
(55)/(55+83)                                #determine Recall                               #0.3985507

require(class)                            #library to impute knn analysis 
KNN_pred=knn(df,df2,df$target,k=1)          #predict knn values
conf_matrix=table(KNN_pred,df$target)       #build confusion matrix
sum(diag(conf_matrix))/nrow(df2)               #determine accuracy                     # 0.8733333
conf_matrix
(12)/(131+12)                                               #determine False Negative Rate  #0.07801418
(130)/(130+11)                                            #determine Recall                         # 0.9219858

require(e1071)                                    #library to perform Naive Bayes Analysis
NB_model=naiveBayes(target~.,data=df)           #takes train data
NB_predict=predict(NB_model,df2,type = 'class')    #predict test data
NB_predict
conf=table(observed=df[,1],predicted=NB_predict)       #build matrix
caret::confusionMatrix(conf)                                          #print confusion matrix
1/(137+1)                              #determine FNR                        #0.007246377
(137)/(137+1)                       #determine Recall                      #0.9927536
137/(137+1+12+0)                #determine Accuracy                #0.9133333
df2$target=NB_predict                     #save the predicted values in test set-target variable. 
