
col_names=names(clean_test)
clean_test[col_names]<-lapply(clean_test[col_names],factor)
#########alternate approach ######
#trans_names=names(clean_test)
trans_names=common
#trans_names<-trans_names[!trans_names %in% c("SalePrice")]
trans_names<-trans_names[!trans_names %in% c("sp_bins")]
trans_names1 <- paste(trans_names, collapse = "+")
####Random forest for feature selection ###
#rf.formulas <- as.formula(paste("SalePrice", trans_names1, sep = " ~ "))
rf.formulas <- as.formula(paste("sp_bins", trans_names1, sep = " ~ "))
set.seed(1)
trans.rf_test<- randomForest(rf.formulas,clean_test, importance=TRUE, ntree=200)

predicted_response <- predict(trans.rf1 ,clean_test)

trans_names=common
trans_names<-trans_names[!trans_names %in% c("sp_bins")]
trans_names1 <- paste(trans_names, collapse = "+")
rf.formulas <- as.formula(paste("sp_bins", trans_names1, sep = " ~ "))
set.seed(1)
trans.rf<- randomForest(rf.formulas,clean_test, importance=TRUE, ntree=200)



bus_bins-----BsmtUnfSF
yr_bins-----YearRemodAdd
yb_bins-----YearBuilt
lf_bins-----LotFrontage
ops_bins----OpenPorchSF
x2fs_bins---X2ndFlrSF
wds_bins----WoodDeckSF
tbsf_bins---TotalBsmtSF
mva_bins----MasVnrArea
bfs1_bins---BsmtFinSF1
Neighborhood
OverallQual
MoSold
TotRmsAbvGrd
Exterior1st
BsmtFinType1
MSSubClass
GarageCars
OverallCond
BedroomAbvGr
FireplaceQu
GarageFinish
FullBath
BsmtExposure
BsmtQual
Fireplaces
HeatingQC
HouseStyle
GarageType
KitchenQual
BsmtFullBath
HalfBath
ExterQual
MasVnrType
SaleCondition
Foundation
#Dummify 
TotRmsAbvGrd
BsmtFinType1
MSSubClass
Exterior1st
Neighborhood
GarageFinish
Foundation
SaleCondition
MasVnrType
GarageType
HouseStyle

#Modify
yr_bins---YearRemodAdd
yb_bins---YearBuilt
#MoSold
#Ordinal
FireplaceQu
BsmtQual
BsmtExposure
ExterQual
KitchenQual
HeatingQC

#Nothing to do
OverallQual
GarageCars
OverallCond
BedroomAbvGr
FullBath
Fireplaces
BsmtUnfSF
LotFrontage
OpenPorchSF
X2ndFlrSF
WoodDeckSF
TotalBsmtSF
MasVnrArea
BsmtFinSF1
HalfBath
BsmtFullBath
