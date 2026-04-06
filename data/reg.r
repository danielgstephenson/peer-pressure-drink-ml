cleanData = read.csv('cleanData.csv')

treatment1 = cleanData$treatment == "Feedback"
treatment2 = cleanData$treatment == "No Feedback"
target = cleanData$measure2

mean(target[treatment1])
mean(target[treatment2])

treatment = 1 * treatment1 + 2 * treatment2

t.test(target[treatment1],target[treatment2])

summary(lm(treatment~cleanData$groupSize))
summary(lm(treatment~cleanData$setting))
