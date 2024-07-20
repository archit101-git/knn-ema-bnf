Overview
This repository contains a Pine Script implementation of a k-Nearest Neighbors (kNN) based trading strategy, enhanced with an Exponential Moving Average (EMA) filter. The strategy aims to predict future market moves by analyzing historical data and identifying similar patterns using the kNN algorithm.

Strategy Description
The strategy uses the kNN algorithm to predict the next market move based on historical data collected in arrays. It evaluates the similarity between the current indicator values and historical values to classify the current market state. The prediction is then used to make trading decisions.

Key Features
kNN Algorithm: Utilizes k-nearest neighbors to find similar historical patterns.
EMA Filter: Adds an EMA filter to refine entry signals.
Multiple Indicators: Supports RSI, ROC, CCI, Volume, or a combination of all as input features.
Volatility Filter: Optional filter based on ATR to manage trade conditions.
