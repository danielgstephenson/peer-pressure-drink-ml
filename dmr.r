outcomeData = read.csv('output/outcome.csv')
treatmentData = read.csv('output/treatment.csv')

summary(lm(outcome ~ output,data=outcomeData))
summary(lm(treatment ~ output,data=treatmentData))

outcome = outcomeData$outcome
treatment = treatmentData$treatment
summary(lm(outcome ~ treatment))

outcomeResidual = outcomeData$outcome - outcomeData$output
treatmentResidual = treatmentData$treatment - treatmentData$output
summary(lm(outcomeResidual ~ treatmentResidual))
