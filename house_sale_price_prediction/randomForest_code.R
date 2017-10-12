
############## Binning #############
train=clean_train

la_bins=cut(unclass(train$LotArea),20,include.lowest=T)
bfs1_bins=cut(unclass(train$BsmtFinSF1),20,include.lowest=T)
mva_bins=cut(unclass(train$MasVnrArea),20,include.lowest=T)
bfs2_bins=cut(unclass(train$BsmtFinSF2),20,include.lowest=T)
bus_bins=cut(unclass(train$BsmtUnfSF),20,include.lowest=T)
tbsf_bins=cut(unclass(train$TotalBsmtSF),20,include.lowest=T)
#x1fs_bins=cut(unclass(train$X1stFlrSF),20,include.lowest=T)
x2fs_bins=cut(unclass(train$X2ndFlrSF),20,include.lowest=T)
#gla_bins=cut(unclass(train$GrLivArea),20,include.lowest=T)
#gyb_bins=cut(unclass(train$GarageYrBlt),20,include.lowest=T)
#ga_bins=cut(unclass(train$GarageArea),20,include.lowest=T)
wds_bins=cut(unclass(train$WoodDeckSF),20,include.lowest=T)
ops_bins=cut(unclass(train$OpenPorchSF),20,include.lowest=T)
ep_bins=cut(unclass(train$EnclosedPorch),20,include.lowest=T)
scrnp_bins=cut(unclass(train$ScreenPorch),20,include.lowest=T)
X3_bins=cut(unclass(train$X3SsnPorch),20,include.lowest=T)
yb_bins=cut(unclass(train$YearBuilt) ,20,include.lowest=T)
lf_bins=cut(unclass(train$LotFrontage) ,20,include.lowest=T)
yr_bins=cut(unclass(train$YearRemodAdd) ,20,include.lowest=T)

sp_bins=as.factor(sp_bins)
train$sp_bins=sp_bins
la_bins=as.factor(la_bins)
train$la_bins = la_bins
bfs1_bins=as.factor(bfs1_bins)
train$bfs1_bins=bfs1_bins
mva_bins=as.factor(mva_bins)
train$mva_bins=mva_bins
bfs2_bins=as.factor(bfs2_bins)
train$bfs2_bins=bfs2_bins
bus_bins=as.factor(bus_bins)
train$bus_bins=bus_bins
tbsf_bins=as.factor(tbsf_bins)
train$tbsf_bins=tbsf_bins
#x1fs_bins=as.factor(x1fs_bins)
#train$x1fs_bins=x1fs_bins
x2fs_bins=as.factor(x2fs_bins)
train$x2fs_bins=x2fs_bins
#gla_bins=as.factor(gla_bins)
#train$gla_bins=gla_bins
#gyb_bins=as.vector(gyb_bins)
#train$gyb_bins=gyb_bins
#ga_bins=as.factor(ga_bins)
#train$ga_bins=ga_bins
wds_bins=as.factor(wds_bins)
train$wds_bins=wds_bins
ops_bins=as.factor(ops_bins)
train$ops_bins=ops_bins
ep_bins=as.factor(ep_bins)
train$ep_bins=ep_bins
scrnp_bins=as.factor(scrnp_bins)
train$scrnp_bins=scrnp_bins
X3_bins=as.factor(X3_bins)
train$X3_bins=X3_bins
yb_bins = as.factor(yb_bins)
train$yb_bins = yb_bins
lf_bins = as.factor(lf_bins)
train$lf_bins = lf_bins
yr_bins = as.factor(yr_bins)
train$yr_bins = yr_bins

### For random forest
train$SalePrice=NULL
train$X3SsnPorch=NULL
train$LotArea=NULL
train$BsmtFinSF1 =NULL
train$MasVnrArea =NULL
train$BsmtFinSF2 =NULL
train$BsmtUnfSF =NULL
train$TotalBsmtSF =NULL
#train$X1stFlrSF =NULL
train$X2ndFlrSF =NULL
#train$GrLivArea =NULL
#train$GarageYrBlt =NULL
#train$GarageArea =NULL
train$WoodDeckSF =NULL
train$OpenPorchSF =NULL
train$ScreenPorch =NULL
train$EnclosedPorch = NULL
train$YearBuilt =NULL
train$LotFrontage = NULL
train$YearRemodAdd =NULL

##### Columns with NAs #######
NAColumns(train)

sp_bins <- train$sp_bins
#train$sp_bins <- NULL
train$Id <- NULL
trnsfrmd_data=train
p=sapply(trnsfrmd_data, function(x){
  typeof(x)=="character"
})
s=subset(p,p==TRUE)
col_names=names(s)
trnsfrmd_data[col_names]<-lapply(trnsfrmd_data[col_names],factor)
trans_names=names(trnsfrmd_data)
#trans_names<-trans_names[!trans_names %in% c("SalePrice")]
trans_names<-trans_names[!trans_names %in% c("sp_bins")]
trans_names1 <- paste(trans_names, collapse = "+")
####Random forest for feature selection ###
#rf.formulas <- as.formula(paste("SalePrice", trans_names1, sep = " ~ "))
rf.formulas <- as.formula(paste("sp_bins", trans_names1, sep = " ~ "))
set.seed(1)
trans.rf<- randomForest(rf.formulas,trnsfrmd_data, importance=TRUE, ntree=200)

# Variable Importance Plot
#varImpPlot(trans.rf,sort = T,main="Variable Importance",n.var=20)

imp1 = importance(trans.rf, type=1, scale=FALSE)
imp2 = importance(trans.rf, type=2, scale=FALSE)
min(imp2)
varImp=data.frame(trans_names,round(imp2,2),round(imp1,4))
x=varImp[with(varImp, order(-imp2)),]
y=varImp[with(varImp, order(-imp1)),]
x=x[1:40,]
y=y[1:40,]
common=intersect(x$trans_names,y$trans_names)
x[common,order]
length(common)

trans_names=common
trans_names<-trans_names[!trans_names %in% c("sp_bins")]
trans_names1 <- paste(trans_names, collapse = "+")
rf1.formulas <- as.formula(paste("sp_bins", trans_names1, sep = " ~ "))
set.seed(1)
trans.rf1<- randomForest(rf1.formulas,trnsfrmd_data, importance=TRUE, ntree=200)
predicted_response <- predict(trans.rf1 ,trnsfrmd_data)
#predicted_response<-as.factor(predicted_response)
sp_bins<-as.factor(sp_bins)
#rcorr(predicted_response,sp_bins)
predicted_response[917]<-"(3.49e+04,3.87e+04]"
predicted_response[969]<-"(3.49e+04,3.87e+04]"
predicted_response[496]<-"(3.49e+04,3.87e+04]"
predicted_salePrice=c()
bin_mean <- function(x){
  mean(cbind(lower = as.numeric( sub("\\((.+),.*", "\\1", x) ), upper = as.numeric( sub("[^,]*,([^]]*)\\]", "\\1", x) )))
}
for(i in 1:length(predicted_response)){
  if(is.na(bin_mean(predicted_response[i]))){
    print(i)
  }
}

predicted_salePrice=sapply(predicted_response, function(x){bin_mean(x)})
predicted_salePrice[c(496,917,969)]=36800
sum(is.na(predicted_salePrice))
rmse(salePrice,predicted_salePrice)
#error=intersect(sp_bins,predicted_response)
NAColumns(clean_test)

####### FOR KNN WE need to normalize the columns ######
#normalize<- function(x) {(x-min(x))/(max(x)-min(x))}
###### dummify the data
clean_train$Id<-NULL
clean_train$sp_bins<-NULL
dummy_data=dummyVars(" ~ .", data = clean_train)
train_dummy_data=data.frame(predict(dummy_data, newdata = clean_train))

library(Hmisc)
res2<-rcorr(as.matrix(train_dummy_data))
#flattenCorrMatrix <- function(cormat, pmat) {
flattenCorrMatrix <- function(cormat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut]
    # p = pmat[ut]
  )
}
p=flattenCorrMatrix(round(res2$r,2))
s=subset(p, p[3]<(-0.7))





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

