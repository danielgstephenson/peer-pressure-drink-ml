mydata = read.csv('output/residuals.csv')
summary(lm(outcome~treatment1+treatment2+treatment3,data=mydata))
