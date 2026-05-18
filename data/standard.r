groupData = read.csv('groupData.csv')
standardData = groupData
n = nrow(standardData)
for(j in 3:11) {
  x = standardData[,j]
  d = x-mean(x)
  msd = sqrt(mean(d^2))
  standardData[,j] = d/msd
}
write.csv(standardData,'standardData.csv',row.names=FALSE)

mspe = rep(0,n)
for(i in 1:n) {
  trainData = standardData[-i,-c(1,2)]
  testData = standardData[i,-c(1,2)]
  reg = lm(outcome  ~ ., data=trainData)
  output = predict(reg,testData)
  mspe[i] = (predict(reg,testData) - testData[1,1])^2
}
mean(mspe)
target = standardData$outcome
mean((target-mean(target))^2)
