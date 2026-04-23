estimates = read.csv('output/estimates.csv')

observations = unique(estimates$observation)
predictions = matrix(0,nrow=length(observations),ncol=4)

for (i in 1:length(observations)) {
  observation = observations[i]
  predictions[observation,1] = mean(estimates$treatment1[estimates$observation==observation])
  predictions[observation,2] = mean(estimates$treatment2[estimates$observation==observation])
  predictions[observation,3] = mean(estimates$treatment3[estimates$observation==observation])
  predictions[observation,4] = mean(estimates$treatment4[estimates$observation==observation])
}

estimatedEffects = predictions[,3] - predictions[,4]

wilcox.test(estimatedEffects)