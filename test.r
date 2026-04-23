originalOutput = read.csv('output/original_test_loss.csv',col.names=c('testLoss','stopTime'))
nullOutput = read.csv('output/null_test_loss.csv',col.names=c('testLoss','stopTime'))
wilcox.test(originalOutput$testLoss,nullOutput$testLoss)