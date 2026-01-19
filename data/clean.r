rawDf = read.table('experiment1.csv', sep='\t', header=TRUE, quote= '')

keepNumericVars = c('measure2','measure1','selected','age','height',
                    'weight','measure1GuessOwn','measure1GuessGroup', 'measure1GuessAll')
keepCategoryVars = c('treatment','setting','female','student')
keepVars = c(keepNumericVars,keepCategoryVars)

df = rawDf[,keepVars]
df[df==""] = NA

df$groupSize = 0*df$measure2
df$peerMean1 = 0*df$measure2
n = dim(df)[1]
for(i in 1:n) {
  g = rawDf$groupId[i]
  p = rawDf$personId[i]
  peers = rawDf$groupId==g & rawDf$personId!=p
  peerMeasures1 = rawDf$measure1[peers]
  df$groupSize[i] = sum(rawDf$groupId==g)
  df$peerMean1[i] = mean(peerMeasures1)
}

df = df[complete.cases(df),]

write.csv(df,'cleanData.csv',row.names=FALSE)

