#########___________CLASSIFICATION AND REGRESSION ENSEMBLE__________##################
library(dismo)
library(gbm)
library(caret)
library(flexclust)
library(boot)
library(ggplot2)
library(pscl)

train=read.csv("C:/Users/deepak/Downloads/Kangaroo_train.csv")
valid=read.csv("C:/Users/deepak/Downloads/Kangaroo_valid.csv")
hold_out=read.csv("C:/Users/deepak/Downloads/Kangaroo_hold.csv")

###_________Loading Gini function_______#########
get.GINI <- function(input            # The name of the input data set
                     ,py              # The name of the column containing the predicted values
                     ,y               # The name of the column containing the actual value
)
{
  set.seed(1)   
  
  # Filter the data
  data <- input
  data$rand.unif <- runif(dim(data)[1])
  
  # Assign weight 1 to all observations
  data$w <- 1
  
  # Rank the data based on predictions
  data <- data[order(data[,py],data[,'rand.unif']),]
  
  test <- data
  
  #Accumulate w to calculate Gini
  for (i in 1:dim(test)[1]){
    if(i==1){test$cumm_w0[i] = 0 + test$w[i]}
    else{
      test$cumm_w0[i] <- test$cumm_w0[i-1] + test$w[i]
      
    }
    
  }
  
  # Calculate Gini
  a <- test[,y]*test$cumm_w0*test$w
  b <- test[,y]*test$w
  
  gini <- 1 - 2 / ( sum(test$w) - 1 )*( sum(test$w) - sum( a ) / sum( b ))
  
  print(paste("Estimated GINI on",round(gini,8),sep=' '))
  
}

#######________Feature Engineering______#########
train$veh_value = train$veh_value*10000
valid$veh_value=valid$veh_value*10000
hold_out$veh_value=hold_out$veh_value*10000


train$logclaimcst0=log((train$claimcst0+1),base = 10)
valid$logclaimcst0=log((valid$claimcst0+1),base = 10)

train$veh_value=log(train$veh_value+1)
valid$veh_value=log(valid$veh_value+1)
hold_out$veh_value=log(hold_out$veh_value+1)


train$veh_type=train$veh_body
levels(train$veh_type)=c("H","H","H","L","H","H","L","H","H","L","H","L","L")
table(train$veh_type,train$clm)

valid$veh_type=valid$veh_body
levels(valid$veh_type)=c("H","H","H","L","H","H","L","H","H","L","H","L","L")
table(valid$veh_type,valid$clm)

hold_out$veh_type=hold_out$veh_body
levels(hold_out$veh_type)=c("H","H","H","L","H","H","L","H","H","L","H","L","L")
table(hold_out$veh_type,hold_out$clm)

levels(train$area)=c("A-D-E","B-C-F","B-C-F","A-D-E","A-D-E","B-C-F")
levels(valid$area)=c("A-D-E","B-C-F","B-C-F","A-D-E","A-D-E","B-C-F")
levels(hold_out$area)=c("A-D-E","B-C-F","B-C-F","A-D-E","A-D-E","B-C-F")

train$agecat=as.factor(train$agecat)
str(train$agecat)
levels(train$agecat)=c("1","2-3-4","2-3-4","2-3-4","5-6","5-6")
table(train$agecat)


valid$agecat=as.factor(valid$agecat)
str(valid$agecat)
levels(valid$agecat)=c("1","2-3-4","2-3-4","2-3-4","5-6","5-6")
table(valid$agecat)

hold_out$agecat=as.factor(hold_out$agecat)
str(hold_out$agecat)
levels(hold_out$agecat)=c("1","2-3-4","2-3-4","2-3-4","5-6","5-6")
table(hold_out$agecat)

train$veh_age=factor(train$veh_age)
str(train)
valid$veh_age=factor(valid$veh_age)
str(valid)
hold_out$veh_age=factor(hold_out$veh_age)
str(hold_out)

levels(train$veh_age)=c("1","2","3-4","3-4")
levels(valid$veh_age)=c("1","2","3-4","3-4")
levels(hold_out$veh_age)=c("1","2","3-4","3-4")


str(train)
str(valid)
str(hold_out)


dmytrain=dummyVars("~id+claimcst0+veh_value+exposure+veh_age+gender+area+agecat+numclaims+logclaimcst0+veh_type",data=train,fullRank = T)
trsf_train <- data.frame(predict(dmytrain, newdata = train))
head(trsf_train)

dmyvalid=dummyVars("~id+claimcst0+veh_value+exposure+veh_age+gender+area+agecat+numclaims+logclaimcst0+veh_type",data=valid,fullRank = T)
trsf_valid <- data.frame(predict(dmyvalid, newdata = valid))
head(trsf_valid)

dmyhold_out=dummyVars("~id+veh_value+exposure+veh_age+gender+area+agecat+veh_type",data=hold_out,fullRank = T)
trsf_hold_out <- data.frame(predict(dmyhold_out, newdata = hold_out))
head(trsf_hold_out)


#--------------------CREATING INTERACTIONS OF VARIOUS VARIABLES WITH EXPOSURE-------------------


trsf_train$int_exposure_vehage=trsf_train$exposure*trsf_train$veh_age.2
trsf_valid$int_exposure_vehage=trsf_valid$exposure*trsf_valid$veh_age.2
trsf_hold_out$int_exposure_vehage=trsf_hold_out$exposure*trsf_hold_out$veh_age.2


trsf_train$int_exposure_vehval=trsf_train$exposure*trsf_train$veh_value
trsf_valid$int_exposure_vehval=trsf_valid$exposure*trsf_valid$veh_value
trsf_hold_out$int_exposure_vehval=trsf_hold_out$exposure*trsf_hold_out$veh_value



#_____Combining both datasets______##
combined_data=data.frame()
combined_data=rbind(trsf_train,trsf_valid)
trsf_train = combined_data


# Zero-Inflated Poisson Regression 
zeroinflated = zeroinfl(numclaims~veh_value+exposure+veh_age.2+veh_age.3.4+gender.M+area.B.C.F+agecat.2.3.4+agecat.5.6+veh_type.L+int_exposure_vehage| exposure+veh_age.2+veh_age.3.4+gender.M+area.B.C.F+agecat.2.3.4+agecat.5.6+veh_type.L+int_exposure_vehage, data = trsf_train)
summary(zeroinflated)

###_______Model builing on training dataset to predict numclaims______###
trsf_train$pred3 = predict(zeroinflated,trsf_train)

###_______Predicting numclaims on hold_out dataset from the above model_____#####
trsf_hold_out$pred3= predict(zeroinflated,trsf_hold_out)

###______Gradient boosting modeling with significant variables along with predicted numclaims from Zero-Inflated Poisson model
gbm_reg=gbm.step(trsf_train,gbm.x=c("veh_value","exposure","veh_age.2","veh_age.3.4","gender.M","area.B.C.F","agecat.2.3.4","agecat.5.6","veh_type.L","int_exposure_vehage","pred3"),
                 gbm.y=c("logclaimcst0"),
                 family="gaussian",
                 n.trees=50,
                 tree.complexity=3,
                 bag.fraction=0.5,
                 learning.rate=0.01)

summary(gbm_reg)
hold_gbm_reg<-predict.gbm(gbm_reg,trsf_hold_out,n.trees=gbm_reg$gbm.call$best.trees,type="response")
trsf_hold_out$pred1 = hold_gbm_reg

###____Writing the predictions_____###
submission=data.frame(id=trsf_hold_out$id,claimst0=trsf_hold_out$pred1)
str(submission)
write.csv(submission,"C:/Users/deepak/submission1.csv",row.names=F)