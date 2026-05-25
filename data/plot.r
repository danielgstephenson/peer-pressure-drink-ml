standardData = read.csv('standardData.csv')

outcome = standardData$outcome
treatment = standardData$treatment
setting1 = standardData$setting1
setting2 = standardData$setting2
setting3 = standardData$setting3
summary(lm(outcome ~ treatment))
summary(lm(outcome ~ treatment + setting1 + setting2 + setting3))

hist(standardData$outcome[standardData$treatment==0],breaks=10,probability=TRUE,ylim=c(0,1),xlim=c(-2,2))
hist(standardData$outcome[standardData$treatment==1],breaks=10,probability=TRUE,ylim=c(0,1),xlim=c(-2,2))

mean(outcome[treatment==0])
mean(outcome[treatment==1])
