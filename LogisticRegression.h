#pragma once

#include <iostream>
#include <vector>
#include <cmath>

class LogisticRegression
{
public:
	// Constructor
	LogisticRegression(const std::vector<std::vector<float>>& features, const std::vector<bool>& targets, const int& numOfFeatures, const int& numOfDataPoints);

	// Public methods
	float Predict(std::vector<float> input, const bool test = false) const;
	void Train(const float& learningRate, const int maxIterations = 40000);
	std::vector<float> GetParameters() const;

private:
	// Private methods
	void Scale();
	float Loss() const;
	std::vector<float> CoeffGradient() const;
	float InterceptGradient() const;
	std::vector<float> GetMean() const;
	std::vector<float> GetStandardDeviation(const std::vector<float>& mean) const;

	// Data members
	std::vector<std::vector<float>> features;
	std::vector<bool> targets;

	// Model parameters
	std::vector<float> coefficients;
	float intercept;

	// Variables
	int numOfFeatures;
	int numOfDataPoints;
	std::vector<float> mean;
	std::vector<float> standardDeviation;
};

