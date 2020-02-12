# Interval for classification accuracy

#common confidence intervals
#    1.64 (90%)
#    1.96 (95%)
#    2.33 (98%)
#    2.58 (99%)


# for 50 samples
# binomial confidence interval
from math import sqrt
interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 50)
print('%.3f' % interval)

# for 100 samples
# binomial confidence interval
interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 100)
print('%.3f' % interval)

# for a dataset of 100 samples; 95% confidence interval with a significance of 0.05
from statsmodels.stats.proportion import proportion_confint
lower, upper = proportion_confint(88, 100, 0.05)
print('lower=%.3f, upper=%.3f' % (lower, upper))
