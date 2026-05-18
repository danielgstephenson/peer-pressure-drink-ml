outcomeData = read.csv('output/outcome.csv')
treatmentData = read.csv('output/treatment.csv')
standardData = read.csv('data/standardData.csv')

plot(outcomeData$outcome, outcomeData$output)
hist(treatmentData$treatment,prob=TRUE)
hist(treatmentData$output,prob=TRUE)
mean(treatmentData$output[treatmentData$treatment==0])
mean(treatmentData$output[treatmentData$treatment==1])

summary(lm(outcome ~ output,data=outcomeData))
summary(lm(treatment ~ output,data=treatmentData))

mean(standardData$sd1[standardData$treatment==0])
mean(standardData$sd1[standardData$treatment==1])

outcome = standardData$outcome
treatment = standardData$treatment
setting1 = standardData$setting1
setting2 = standardData$setting2
setting3 = standardData$setting3
summary(lm(outcome ~ treatment))
summary(lm(outcome ~ treatment + setting1 + setting2 + setting3))

outcomeResidual = outcomeData$outcome - outcomeData$output
treatmentResidual = treatmentData$treatment - treatmentData$output
summary(lm(outcomeResidual ~ treatmentResidual))
