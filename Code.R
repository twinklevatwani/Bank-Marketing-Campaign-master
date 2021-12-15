## Part a) Data Loading of Bank data
df<-read.csv("F:\\Aegis\\Machine Learning\\Topic 8\\bank.csv",sep = ";")
head(df)
View(df)
summary(df)
str(df)
colnames(df)
table(df$default)

table(df$pdays,df$previous)
table(df$y,df$previous)

## Part b) Appropriate Methods for Significant Variables

## For significant variables:
## for numerical variables
library(corrgram)
dn<-df[c(1,6,10,12:15,17)]
dn$y<-as.numeric(ifelse(dn$y=="no",0,1))
cor(dn,method = "pearson")

## for categorical variables
library(devtools)
library(woe)
dd<-df[-c(1,6,10,12:15)]
iv<-iv.mult(dd,"y",TRUE)
iv.plot.summary(iv)

## from the dataframe, least significant variables are default, from iv plot(for categorical)
## and day,balance and age from correlation(numerical)

## Using Random Forest to verify and determine significant variables.
library(randomForest)
rf<-randomForest(y~.,data = df)
rf$importance
colnames(df)
## from the model importance , removing the columns: default, loan, housing, contact, marital, previous and education.

df<-df[!names(df)%in% c("default","loan","housing","contact","marital","previous","education")]

##normalization of the numerical variables.
ind<-c("age","balance","day","duration","campaign","pdays")
df[ind]<-as.data.frame(lapply(df[ind],scale))


## Part c) Dividing data into Development(train) and Validation(test) data.
library(caTools)
set.seed(1)
x<-sample.split(df$y,SplitRatio = 0.7)
train<-subset(df,x==T)
test<-subset(df,x==F)


## Part d) SVM model with linear kernel and Accuracy with Test data.
library(e1071)
model=svm(y~.,data = train,kernel="linear")
model
summary(model)
results<-predict(model,test)
#precision<-sum(as.integer(results) & as.integer(test$y))/sum(as.integer(results))
class(results)

# accuracy of SVM linear kernel
table(results,test$y)
mean(results==test$y)

library(caret)
?precision()

## Part e) SVM with radial kernel, Tuning the model, and Accuracy with Test data.
## tuning the radial svm model.
model_tune<-tune(svm,y~.,data = train,kernel="radial",ranges = list(cost=c(0.001,0.01,0.1,1,5),gamma=c(0.001,0.01,0.1,1,5)))
summary(model_tune)
bestmodel= model_tune$best.model
summary(bestmodel)

## best svm model has cost 5 and gamma 0.01

## SVM with radial kernel
model_r<-svm(y~.,data = train,kernel="radial",cost=5,gamma=0.01)
model_r
results_r<-predict(model_r,test)

# accuracy of SVM radial kernel
table(results_r,test$y)
mean(results_r==test$y)


## Part f) Naive Bayes Algorithm , and compare results with SVM.
### Naive-Bayes Model
model_nb<-naiveBayes(y~.,data = train)
model_nb
results_nb<-predict(model_nb,test)
table(results_nb,test$y)
mean(results_nb==test$y)


## The SVM with radial kernel accuracy comes out to be 0.8913, whereas Naive-Bayes accuracy
## is around 0.8716. So the tuned SVM with radial kernel performs slightly better than the 
## Naive- Bayes model, with the default parameters.
