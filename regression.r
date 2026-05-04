residualData = read.csv('output/residuals.csv')
outcomeRes = residualData$outcome
treatmentRes1 = residualData$treatment1
treatmentRes2 = residualData$treatment2
treatmentRes3 = residualData$treatment3

cateData = read.csv('output/cate.csv')
cate1 = cateData$cate1
cate2 = cateData$cate2
cate3 = cateData$cate3
x1 = cate1 * treatmentRes1
x2 = cate2 * treatmentRes2
x3 = cate3 * treatmentRes3

mean(cate1)
mean(cate2)
mean(cate3)
cateReg = lm(outcomeRes~x1+x2+x3)
summary(cateReg)

residualReg = lm(outcomeRes~treatmentRes1+treatmentRes2+treatmentRes3)
summary(residualReg)

