cleanData = read.csv('cleanData.csv')
groupIds = unique(cleanData$groupId)
groupSize = 0*groupIds
for(g in 1:length(groupIds)) {
  selection = cleanData$groupId == groupIds[g]
  groupSize[g] = sum(selection)
}
group = groupIds[groupSize>1]
groupSize = groupSize[groupSize>1]
treatmentFactor = as.factor(cleanData$treatment)
settings = unique(cleanData$setting)
outcome = 0*group
mean1 = 0*group
sd1 = 0*group
age = 0*group
height = 0*group
weight = 0*group
female = 0*group
student = 0*group
setting1 = 0*group
setting2 = 0*group
setting3 = 0*group
treatment = 0*group

for(g in 1:length(group)) {
  selection = cleanData$groupId == group[g]
  measure1 = cleanData$measure1[selection]
  measure2 = cleanData$measure2[selection]
  mean1[g] = mean(measure1)
  mean2 = mean(measure2)
  relativeMeasure1 = measure1 - mean1[g]
  relativeMeasure2 = measure2 - mean2
  dRelativeMeasure = relativeMeasure2 - relativeMeasure1
  outcome[g] = mean(sign(relativeMeasure1*dRelativeMeasure)) # divergence
  sd1[g] = sd(measure1)
  age[g] = mean(cleanData$age[selection])
  height[g] = mean(cleanData$height[selection])
  weight[g] = mean(cleanData$weight[selection])
  female[g] = mean(cleanData$female[selection]=='Female')
  student[g] = mean(cleanData$student[selection]=='Student')
  setting1[g] = cleanData$setting[selection][1] == settings[1]
  setting2[g] = cleanData$setting[selection][1] == settings[2]
  setting3[g] = cleanData$setting[selection][1] == settings[3]
  treatment[g] = treatmentFactor[selection][1] != 'No Feedback'
}

groupData = data.frame(
  group,treatment,outcome,
  mean1,sd1,groupSize,age,height,weight,
  female,student,setting1,setting2,setting3
)
write.csv(groupData,'groupData.csv',row.names=FALSE)
