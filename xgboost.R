######## LOAD ALL THE NECESSARY LIBRARIES #######

library(readr)

library(randomForest)

library(data.table)

library(bit64)

library(xgboost)

library(lubridate)

library(caret)

library(glmnet)




set.seed(1)



setwd("../download/AV")



###### Read data files #########

train <- fread("train.csv",header = T,sep = ',',stringsAsFactors=TRUE)

train <-as.data.frame(train)


test <- fread("test.csv",header = T,sep = ',',stringsAsFactors=TRUE)

test <-as.data.frame(test)


alcohol <- fread("alcohol.csv",header = T,sep = ',',stringsAsFactors=TRUE)

alcohol <-as.data.frame(alcohol)


gc()



trainexH<-train

trainexH$Happy=NULL


combined<-rbind(trainexH,test)


combined <- merge(combined,alcohol, by="ID")

combined$ID = NULL



################# UNKNOWN FIELDS ANALYSIS ##################
#
## "Var1", "Var2"

combined$Var1 <- as.numeric(factor(combined$Var1)
)

################# WORK PROFILE ANALYSIS ##################
##

combined$WorkStatus <- as.numeric(factor(combined$WorkStatus))
combined$Unemployed10 <- as.numeric(factor(combined$Unemployed10))

combined$income[combined$income=="lt $1000"] = 500
res <- sapply(strsplit(combined$income, " "), "[[", 1)
combined$income <- as.numeric(gsub("\\$"," ",res))


################# FAMILY PROFILE ANALYSIS ##################
##
# RELIGION, TV, GENDER, RESIDENCE REGION

combined$children <- combined$babies + combined$preteen + combined$teens
combined$Divorce <- as.numeric(factor(combined$Divorce))
combined$Widowed <- as.numeric(factor(combined$Widowed))
combined$Alcohol_Consumption <- as.numeric(factor(combined$Alcohol_Consumption))

################# OTHER DATA PROFILE ANALYSIS ##################
##
# RELIGION, TV, GENDER, RESIDENCE REGION

combined$Gender <- as.numeric(factor(combined$Gender))
combined$Residence_Region <- as.numeric(factor(combined$Residence_Region))
combined$Engagement_Religion <- as.numeric(factor(combined$Engagement_Religion))

################# CREATE ADDITIONAL VARIABLES ##################
##
# GENDER + UNEMPLOYED, FAMILY + UNEMPLOYED INCOMER SPLIT FOR FAMILY

combined$gend_unemp <- as.numeric(paste(combined$Unemployed10,combined$Gender,  sep = ""))
combined$family_unemp <- as.numeric(paste(combined$Unemployed10, combined$children, sep = ""))

combined$family_income <- combined$income * combined$children

################# MISSING DATA ANALYSIS ##################
##
# TREAT NA WITH -1

#combined$Var1[is.na(combined$Var1)
] <- median(combined$Var1,na.rm=TRUE)
combined$Var2[is.na(combined$Var2)] = median(combined$Var2,na.rm=TRUE)
#combined$WorkStatus[is.na(combined$WorkStatus)] = median(combined$WorkStatus,na.rm=TRUE)
combined$Score[is.na(combined$Score)
] <- median(combined$Score,na.rm=TRUE)
combined$income[is.na(combined$income)] = median(combined$income,na.rm=TRUE)
combined$Education[is.na(combined$Education)] = median(combined$Education,na.rm=TRUE)
#combined$Unemployed10[is.na(combined$Unemployed10)] = median(combined$Unemployed10,na.rm=TRUE)
combined$babies[is.na(combined$babies)] = median(combined$babies,na.rm=TRUE)
combined$preteen[is.na(combined$preteen)] = median(combined$preteen,na.rm=TRUE)
combined$teens[is.na(combined$teens)] = median(combined$teens,na.rm=TRUE)
#combined$Divorce[is.na(combined$Divorce)] = median(combined$Divorce,na.rm=TRUE)
#combined$Widowed[is.na(combined$Widowed)] = median(combined$Widowed,na.rm=TRUE)
#combined$Engagement_Religion[is.na(combined$Engagement_Religion)] = median(combined$Engagement_Religion,na.rm=TRUE)
combined$TVhours[is.na(combined$TVhours)] = median(combined$TVhours,na.rm=TRUE)
#combined$Residence_Region[is.na(combined$Residence_Region)] = median(combined$Residence_Region,na.rm=TRUE)
#combined$Gender[is.na(combined$Gender)] = median(combined$Gender,na.rm=TRUE)
#combined$Alcohol_Consumption <- as.numeric(factor(combined$Alcohol_Consumption))

#combined$children[is.na(combined$children)] = median(combined$children,na.rm=TRUE)
#combined$family_income[is.na(combined$family_income)] = median(combined$family_income,na.rm=TRUE)
#combined$gend_unemp[is.na(combined$gend_unemp)] = median(combined$gend_unemp,na.rm=TRUE)
#combined$family_unemp[is.na(combined$family_unemp)] = median(combined$family_unemp,na.rm=TRUE)

################# REMOVE UNNECESSARY COLUMNS ##################
##
#combined$babies = NULL
#combined$preteen= NULL
#combined$teens= NULL
#combined$Var1 = NULL
#combined$Var2 = NULL
#combined$WorkStatus = NULL
#combined$Score = NULL
#combined$income = NULL
#combined$Education = NULL
#combined$Unemployed10 = NULL
#combined$Divorce = NULL
#combined$Widowed = NULL
#combined$TVhours = NULL
#combined$Residence_Region = NULL
#combined$Gender = NULL










### PREPARE DATA  ####

trainP<-head(combined,nrow(train))

testP<-tail(combined,nrow(test))


trainP$Happy<-as.factor(train$Happy)
trainP$Happy<-as.numeric(trainP$Happy)-1

feature.names <- names(trainP)[1:(ncol(trainP))-1]


################ XGB MODELLING #########################


cat(".......Training XGBOOST.........\n")



param <- list("objective" = "multi:softmax",    # multiclass classification 
              "num_class" = 3,    # number of classes 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 6,    # maximum depth of tree 
              "eta" = 0.05,   # step size shrinkage 
              "gamma" = 0    # minimum loss reduction 
              )

cust_eval <- function(actual, predicted)
{
  labels <- getinfo(predicted, "label")  
  err <- custom_eval_func(as.numeric(labels),as.numeric(actual))
  return(list(metric = "Custom", value = err))
}

custom_eval_func <- function(actual, predicted)
{
  if (all(actual %in% c(2,1,0)) & all(predicted %in% c(2,1,0)))
  {
    actual <- ifelse(actual == 2, 15, ifelse(actual == 1, 10, 5))
    predicted <- ifelse(predicted == 2, 15, ifelse(predicted == 1, 10, 5))
    diff <- actual - predicted

   score <- (length(diff[diff == 0])*50 + length(diff[diff == 5])*10 + length(diff[diff == 10])*5 + length(diff[diff == -5])*-5 + length(diff[diff == -10])*-10)/(50*length(actual))   
  }
   return(score)
}
​


# k-fold cross validation, with timing



nround.cv = 200

system.time( xgb_check <- xgb.cv(param=param, data=data.matrix(trainP[,feature.names]), label=trainP$Happy, 

              feval=cust_eval,nfold=6, nrounds=nround.cv,prediction=TRUE,missing=NaN, verbose=TRUE) )



tail(xgb_check$dt)

max.merror.idx <- which.max(xgb_check$dt[,test.Custom.mean])
max.merror.idx

# get CV's prediction decoding
pred.cv = matrix(xgb_check$pred)
# confusion matrix
confusionMatrix(factor(trainP$Happy+1), factor(pred.cv+1))

################ TRAIN USING TRAINING DATA #########################



system.time( xgb_chk <- xgboost(param=param, feval=cust_eval,data=data.matrix(trainP[,feature.names]), label=trainP$Happy, missing=NaN,nrounds=max.merror.idx,verbose=TRUE) )



gc()





################ APPLY MODEL ON TEST DATA #########################



xgb_val <- data.frame(ID=test$ID)

xgb_val$Happy <- NA


for(rows in split(1:nrow(testP), ceiling((1:nrow(testP))/1000))) 
{

	xgb_val[rows, "Happy"] <- predict(xgb_chk, data.matrix(testP[rows,feature.names]),missing=NaN)

}



gc()

xgb_val$Happy[xgb_val$Happy==0]="Not Happy"
xgb_val$Happy[xgb_val$Happy==1]="Pretty Happy"
xgb_val$Happy[xgb_val$Happy==2]="Very Happy"




################ SIMPLE ENSEMBLE #########################



write.csv(xgb_val, "sub_xgb_1.csv", row.names=FALSE)

