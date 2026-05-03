mydata = read.csv('output/baseline.csv')
plot(mydata$target,mydata$output)
abline(a=0,b=1)