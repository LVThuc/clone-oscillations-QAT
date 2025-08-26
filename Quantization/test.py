from tkinter import Y
from sympy import Range
from Quantization.range_estimators import RangeEstimator, RangeEstimatorBase

X = RangeEstimator.CurrentMinMax
Y = RangeEstimator.RunningMinMax.cls
Z = RangeEstimator.CurrentMinMax()
print(X)
print(Y)
print(Z)