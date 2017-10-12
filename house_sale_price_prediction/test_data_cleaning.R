library(caret) ###converting categorical to numerical
library(mlbench)
library(randomForest) ### classification algorithm
library(dplyr)
library(ggplot2)
library(data.table)
library(MASS) 
library(Metrics)
library(corrplot)
library(lars)
library(ggplot2)
library(xgboost)
library(Matrix)
library(methods)


setwd("C:/Users/PAPINENI'S/Desktop/NowOn/Fall2016/Data Mining/HOUSING_PROJECT")
########## Reading train.csv ######################
########***Replace your file path below***#######
train_orig <- read.csv("train.csv", stringsAsFactors = F)
train_salePrice<-train_orig$SalePrice
train_Ids<-train_orig$Id
################ Test Data################
test_orig= read.csv("test.csv", stringsAsFactors = F)
test_Ids<-test_orig$Id
train_orig$SalePrice<-NULL
################ Combining the data #######
tot_orig=rbind(train_orig, test_orig)
############### Replacing MEaningful NA #################
tot_orig$Alley[is.na(tot_orig$Alley)]="NoAlley"
tot_orig$MiscFeature[is.na(tot_orig$MiscFeature)]="None"
tot_orig$Fence[is.na(tot_orig$Fence)]="NoFence"
tot_orig$PoolQC[is.na(tot_orig$PoolQC)]="NoPool"
tot_orig$GarageCond[is.na(tot_orig$GarageCond)]="NoGarage"
tot_orig$GarageQual[is.na(tot_orig$GarageQual)]="NoGarage"
tot_orig$GarageFinish[is.na(tot_orig$GarageFinish)]="NoGarage"
tot_orig$GarageType[is.na(tot_orig$GarageType)]="NoGarage"
tot_orig$FireplaceQu[is.na(tot_orig$FireplaceQu)]="NoFireplace"
tot_orig$BsmtFinType2[is.na(tot_orig$BsmtFinType2)]="NoBasement"
tot_orig$BsmtFinType1[is.na(tot_orig$BsmtFinType1)]="NoBasement"
tot_orig$BsmtExposure[is.na(tot_orig$BsmtExposure)]="NoBasement"
tot_orig$BsmtCond[is.na(tot_orig$BsmtCond)]="NoBasement"
tot_orig$BsmtQual[is.na(tot_orig$BsmtQual)]="NoBasement"
##### Columns with NAs #######
NAColumns <- function(data_test){
  s=sapply(data_test, function(x) sum(is.na(x)))
  subset(s,s>0)
}
NAColumns(tot_orig)

##### Dropping GarageYrBlt as it's highly correlated with other variables ######
tot_orig$GarageYrBlt <- NULL
###### After Correlation Results #########
tot_orig$X1stFlrSF <-NULL
tot_orig$GarageArea <-NULL
tot_orig$GrLivArea <- NULL
tot_orig$Exterior2nd <- NULL

clean_train=data.frame(tot_orig[1:1460,])
clean_test=data.frame(tot_orig[1461:2919,])

clean_train$SalePrice <- train_salePrice
##### Binning SalePrice ######
min_sp=min(clean_train$SalePrice)
max_sp=max(clean_train$SalePrice)

#Making 30 bins
f <- function(x) ((min_sp*(1+x)^30)-(max_sp))
bin_width_ratio= uniroot(f, lower=0.01, upper=0.4)$root
bin_breaks = c()
bin_breaks[1] = min(clean_train$SalePrice)
for (i in 1:30) {
  bin_breaks[i+1] <- ceiling(bin_breaks[i]+bin_width_ratio*bin_breaks[i])
}
bin_breaks[1] = min(clean_train$SalePrice)-1

################### test 1 with different bins ####################
sp_bins=cut(unclass(clean_train$SalePrice), bin_breaks, include.lowest=T)
#sp_bins=cut(unclass(train_orig$SalePrice),50,include.lowest=T)
sp_bins=as.vector(sp_bins)
clean_train$sp_bins=sp_bins

###### treating for Missing values#######
###### Electrical #######
p = subset(clean_train, is.na(clean_train$Electrical))
p = subset(clean_train, clean_train$sp_bins==p$sp_bins)
Mode <- function(x) {
  ux <- unique(x)
  ux[is.na(ux)]=0
  ux[which.max(tabulate(match(x, ux)))]
}
electricalMode = Mode(p$Electrical)
clean_train$Electrical=replace(clean_train$Electrical, is.na(clean_train$Electrical), electricalMode)
#train_orig$Electrical=newElectrical
NAColumns(clean_train)

########### MasVnrArea ##########
### imputing with mean based on sp_bins ######
p=subset(clean_train, is.na(clean_train$MasVnrArea))
p$MasVnrArea = sapply(p$sp_bins, function(x){ 
  sp_bins_subset <- subset(clean_train, clean_train$sp_bins==x)
  round(mean(sp_bins_subset$MasVnrArea, na.rm=TRUE))
})
clean_train[match(p$Id,clean_train$Id),] <- p

###### MasVnrType ########
### imputing with mean based on sp_bins ###
p=subset(clean_train, is.na(clean_train$MasVnrType))
p$MasVnrType = sapply(p$sp_bins, function(x){ 
  sp_bins_subset <- subset(clean_train, clean_train$sp_bins==x)
  Mode(sp_bins_subset$MasVnrType)
})
clean_train[match(p$Id,clean_train$Id),] <- p

####### Lot front age #####
####### Imputing values with Class mean #####
p=subset(clean_train, is.na(clean_train$LotFrontage))
p$LotFrontage = sapply(p$sp_bins, function(x){ 
  sp_bins_subset <- subset(clean_train, clean_train$sp_bins==x)
  floor(mean(sp_bins_subset$LotFrontage, na.rm=TRUE))
})
clean_train[match(p$Id,clean_train$Id),] <- p

p=sapply(clean_train, function(x){
  typeof(x)=="character"
})
s=subset(p,p==TRUE)
col_names=names(s)
clean_train[col_names]<-lapply(clean_train[col_names],factor)
clean_train$SalePrice<-train_salePrice
clean_train$sp_bins<-NULL
partition <- createDataPartition(y=clean_train$SalePrice,p=.5,list=F)
training <- clean_train[partition,]
testing <- clean_train[-partition,]

testing_salePrice<-testing$SalePrice
training_salePrice<-training$SalePrice
testing$SalePrice<-NULL
training$SalePrice<-NULL
training_IDs<-training$Id
testing_Ids<-testing$Id
training$Id<-NULL
testing$Id<-NULL
NAColumns(training)
set.seed(1234)
dummy_data=dummyVars(" ~ .", data = training)
train_dummy_data=data.frame(predict(dummy_data, newdata = training))
NAColumns(train_dummy_data)
lr2 = lm(training_salePrice ~., data = train_dummy_data)
set.seed(1234)
dummy_data=dummyVars(" ~ .", data = testing)
test_dummy_data=data.frame(predict(dummy_data, newdata = testing))
predict.train = predict(lr2, newdata = test_dummy_data)
testing$SalePrice = testing_salePrice
testing$PredictSalePrice<-predict.train
rmse(testing$SalePrice,predict.train)
############ Summary #########
clean_train1<-clean_train
clean_train1$Condition1<-NULL
clean_train1$MiscVal<-NULL
clean_train1$YrSold <-NULL
clean_train1$MSZoning <-NULL
clean_train1$MSSubClass <- NULL
clean_train1$Street<-NULL
clean_train1$Alley <- NULL
clean_train1$LotShape <- NULL
clean_train1$Utilities <- NULL
clean_train1$LandContour <- NULL
clean_train1$BldgType<- NULL
clean_train1$RoofStyle <- NULL
clean_train1$BsmtFinType1 <- NULL
clean_train1$BsmtFinType2 <- NULL
clean_train1$TotalBsmtSF <- NULL
clean_train1$Heating <- NULL
clean_train1$CentralAir <- NULL
clean_train1$BsmtHalfBath <- NULL
clean_train1$HalfBath <- NULL
clean_train1$GarageFinish <- NULL
clean_train1$PavedDrive <- NULL
clean_train1$Fence <- NULL
clean_train1$MiscFeature <- NULL
clean_train1$Foundation <- NULL
clean_train1$Electrical <- NULL
clean_train1$EnclosedPorch <- NULL
clean_train1$OpenPorchSF <- NULL
#######################################

clean_train2 <- clean_train1
clean_train2$MasVnrArea <- NULL
clean_train2$LandSlope <- NULL
clean_train2$BsmtCond <- NULL
clean_train2$LowQualFinSF <- NULL
clean_train2$BsmtFullBath <- NULL
clean_train2$FullBath <- NULL
clean_train2$FireplaceQu <- NULL
clean_train2$GarageCond <- NULL

clean_train2$garqual[clean_train2$GarageQual == "Ex"] <- 5
clean_train2$garqual[clean_train2$GarageQual == "Gd"] <- 4
clean_train2$garqual[clean_train2$GarageQual == "TA"] <- 3
clean_train2$garqual[clean_train2$GarageQual == "Fa"] <- 2
clean_train2$garqual[clean_train2$GarageQual == "Po"] <- 1
clean_train2$garqual[clean_train2$GarageQual == "NoGarage"] <- 0
clean_train2$GarageQual <- NULL
clean_train2$pool_good[clean_train2$PoolQC %in% c("Ex")] <- 1
clean_train2$pool_good[!clean_train2$PoolQC %in% c("Ex")] <- 0
clean_train2$PoolQC<-NULL
clean_train2$saleType_NEW[clean_train2$SaleType == "New"] <- 1
clean_train2$saleType_NEW[clean_train2$SaleType != "New"] <- 0

clean_train2$SaleType <- NULL
clean_train2$HeatingQC <- NULL
clean_train2$ExterCond <- NULL
clean_train2$Exterior1st <- NULL
clean_train2$Condition2_2[clean_train2$Condition2 %in% c("PosA", "PosN")] <- 1
clean_train2$Condition2_2[!clean_train2$Condition2 %in% c("PosA", "PosN")] <- 0
clean_train2$Condition2 <- NULL
clean_train2$YearBuilt <- 2016- clean_train2$YearBuilt
clean_train2$YearRemodAdd <- 2016- clean_train2$YearRemodAdd
clean_train2$housefunction[clean_train2$Functional %in% c("Typ", "Mod")] <- 1
clean_train2$housefunction[!clean_train2$Functional %in% c("Typ", "Mod")] <- 0
clean_train2$Functional <- NULL
clean_train2$kitchen_qual[clean_train2$KitchenQual  %in% c("Ex", "Gd")] <- 1
clean_train2$kitchen_qual[!clean_train2$KitchenQual  %in% c("Ex", "Gd")] <- 0
clean_train2$KitchenQual <- NULL
clean_train2$Bsmt_exposure[clean_train2$BsmtExposure=="Gd"] <- 3
clean_train2$Bsmt_exposure[clean_train2$BsmtExposure=="Av"] <- 2
clean_train2$Bsmt_exposure[clean_train2$BsmtExposure=="Mn"] <- 1
clean_train2$Bsmt_exposure[clean_train2$BsmtExposure %in% c("No", "NoBasement")] <- 0
clean_train2$BsmtExposure <- NULL
clean_train2$Bsmt_qual[clean_train2$BsmtQual=="Gd"] <- 3
clean_train2$Bsmt_qual[clean_train2$BsmtQual=="Ex"] <- 4
clean_train2$Bsmt_qual[clean_train2$BsmtQual=="Fa"] <- 2
clean_train2$Bsmt_qual[clean_train2$BsmtQual=="TA"] <- 1
clean_train2$Bsmt_qual[clean_train2$BsmtQual=="NoBasement"] <- 0
clean_train2$BsmtQual <- NULL
clean_train2$MasVnrType <- NULL
clean_train2$Extrnl_qual[clean_train2$ExterQual=="Gd"] <- 3
clean_train2$Extrnl_qual[clean_train2$ExterQual=="Ex"] <- 4
clean_train2$Extrnl_qual[clean_train2$ExterQual=="Fa"] <- 2
clean_train2$Extrnl_qual[clean_train2$ExterQual=="TA"] <- 1
clean_train2$ExterQual <- NULL

clean_train2$Lot_Config[clean_train2$LotConfig=="CulDSac"] <-1
clean_train2$Lot_Config[!clean_train2$LotConfig=="CulDSac"] <- 0
clean_train2$LotConfig <- NULL
clean_train2$X3SsnPorch <- NULL
clean_train2$PoolArea <- NULL



set.seed(1234)
partition <- createDataPartition(y=clean_train2$SalePrice,p=.5,list=F)
training <- clean_train2[partition,]
testing <- clean_train2[-partition,]

testing_salePrice<-testing$SalePrice
training_salePrice<-training$SalePrice
testing$SalePrice<-NULL
training$SalePrice<-NULL

training_IDs<-training$Id
testing_Ids<-testing$Id
training$Id<-NULL
testing$Id<-NULL



##############################################################################################

kitchen_qual= KitchenQual
Bsmt_exposure = BsmtExposure
Condition2_2 = Condition2
saleType_NEW = SaleType
Bsmt_qual = BsmtQual
Extrnl_qual = ExterQual
Lot_Config = LotConfig
pool_good = PoolQC

names_replaced <- c("LotFrontage","LotArea","Neighborhood","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofMatl","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","X2ndFlrSF","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageType","GarageCars","WoodDeckSF","ScreenPorch","MoSold","SaleCondition","GarageQual","Functional","KitchenQual","BsmtExposure","Condition2","SaleType","BsmtQual","ExterQual","LotConfig","PoolQC")
test_columns = NAColumns(clean_test)
selected_variables=intersect(names(test_columns),names_replaced)

s=subset(clean_test, is.na(clean_test$BsmtUnfSF))
p=sapply(s$BsmtQual, function(x){
  replace(x,x=="NoBasement",0)
})
s$BsmtUnfSF=p
clean_test[match(s$Id,clean_test$Id),] <- s

s= subset(clean_test,is.na(clean_test$LotFrontage))
lf<-sapply(s$LotFrontage, function(x){replace(x,is.na(x),mean(clean_test$LotFrontage,na.rm=TRUE))})
s$LotFrontage=lf
clean_test[match(s$Id,clean_test$Id),] <- s

s= subset(clean_test,is.na(clean_test$TotalBsmtSF))
p<-sapply(s$BsmtQual, function(x){replace(x,x=="NoBasement",0)})
s$TotalBsmtSF<-p
clean_test[match(s$Id,clean_test$Id),] <- s

s<-subset(clean_test,is.na(clean_test$GarageCars))
p<-sapply(s$GarageQual,function(x){replace(x,x=="NoGarage",0)})
s$GarageCars<-p
clean_test[match(s$Id,clean_test$Id),] <- s


s= subset(clean_test,is.na(clean_test$KitchenQual))
p<-sapply(s$KitchenQual, function(x){replace(x,is.na(x),Mode(clean_test$KitchenQual))})
s$LotFrontage=p
clean_test[match(s$Id,clean_test$Id),] <- s


nas<-NAColumns(clean_test)
names(nas)
clean_test$YearRemodAdd<-2016-clean_test$YearRemodAdd
clean_test$YearBuilt<-2016-clean_test$YearBuilt

test_columns = NAColumns(clean_test)

diff_columns<-setdiff(names(names(training)),test_data)
test_data$GarageCars<- as.integer(test_data$GarageCars)
test_data$BsmtUnfSF<- as.integer(test_data$BsmtUnfSF)
test_data$TotalBsmtSF<- as.integer(test_data$TotalBsmtSF)
test_data$BsmtUnfSF<- floor(test_data$BsmtUnfSF)
test_data$TotalBsmtSF<- floor(test_data$TotalBsmtSF)

dummy_data=dummyVars(" ~ .", data = test_data)
test_dummy_data=data.frame(predict(dummy_data, newdata = test_data))

NAColumns(train_dummy_data)
######get original data#####
train$YearBuilt<-2016-train$YearBuilt
train$YearRemodAdd<-2016-train$YearRemodAdd

train_data<-train
train_data[,diff_columns]<- NULL
dummy_data=dummyVars(" ~ .", data = train_data)
train_dummy_data=data.frame(predict(dummy_data, newdata = train_data))


lr = lm(train$SalePrice ~., data = train_dummy_data)
predict.train = predict(lr, newdata = train_dummy_data)

train_saleprice=floor(predict.train)
train_ids<-clean_train$Id
test_ids<-clean_test$Id
train_prediction<-data.frame()

predict.Test = predict(lr, newdata = test_dummy_data)
#write.csv(final_prediction,file='C:/Users/manub/Downloads/DM/project/house prices/HP_solution.csv',row.names=F)
final_prediction=data.frame(Id=train_orig$Id, Predicted_SalePrice=predict.train, SalePrice=train_orig$SalePrice)
final_prediction=data.frame(test_ids, predict.train)

rmse(final_prediction$Predi)






