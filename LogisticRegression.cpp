#include "LogisticRegression.h"

// Constructor to initialize the LogisticRegression object with features, targets, and dimensions
LogisticRegression::LogisticRegression(const std::vector<std::vector<float>>& features, const std::vector<bool>& targets, const int& numOfFeatures, const int& numOfDataPoints)
	: features(features), targets(targets), numOfFeatures(numOfFeatures), numOfDataPoints(numOfDataPoints), coefficients(numOfFeatures, 0), intercept(0)
{
	Scale(); // Apply feature scaling upon initialization
}

// Scale the features using Z-Scale Normalization
void LogisticRegression::Scale()
{
	// Calculate mean and standard deviation of features
	mean = GetMean();
	standardDeviation = GetStandardDeviation(mean);

	// Apply Z-score normalization
	for (int i = 0; i < numOfDataPoints; i++) {
		for (int j = 0; j < numOfFeatures; j++) {
			if (standardDeviation[j] != 0) {
				features[i][j] = (features[i][j] - mean[j]) / standardDeviation[j];
			}
		}
	}
}

// Predict the target value for a given input vector
float LogisticRegression::Predict(std::vector<float> input, const bool test) const
{
	if (test) {
		// Feature scaling of input test data
		for (int i = 0; i < numOfFeatures; i++) {
			input[i] = (input[i] - mean[i]) / standardDeviation[i];
		}
	}

	float result = intercept;

	for (int i = 0; i < numOfFeatures; i++) {
		result += input[i] * coefficients[i];
	}

	// Apply sigmoid function
	result = 1.0 / (1.0 + exp(-result));
	return result;
}

// Calculate the Maximum Likelihood Estimation (MLE) loss
float LogisticRegression::Loss() const
{
	float result = 0;

	for (int i = 0; i < numOfDataPoints; i++) {
		float error = targets[i] * std::log(Predict(features[i])) + (1 - targets[i]) * std::log(1 - Predict(features[i]));
		result += error;
	}

	result /= -1.*numOfDataPoints;

	return result;
}

// Calculate the gradient of the cost function with respect to coefficients vector parameter
std::vector<float> LogisticRegression::CoeffGradient() const
{
	std::vector<float> result(numOfFeatures);

	for (int i = 0; i < numOfFeatures; i++) {
		float sum = 0;
		for (int j = 0; j < numOfDataPoints; j++) {
			sum += (Predict(features[j]) - targets[j]) * features[j][i];
		}
		result[i] = sum / numOfDataPoints;
	}

	return result;
}

// Calculate the gradient of the cost function with respect to intercept parameter
float LogisticRegression::InterceptGradient() const
{
	float result = 0;

	for (int i = 0; i < numOfDataPoints; i++) {
		result += Predict(features[i]) - targets[i];
	}
	result /= numOfDataPoints;

	return result;
}

// Calculate the mean of the input features for feature scaling
std::vector<float> LogisticRegression::GetMean() const
{
	std::vector<float> mean(numOfFeatures, 0);

	for (int i = 0; i < numOfFeatures; i++) {
		for (int j = 0; j < numOfDataPoints; j++) {
			mean[i] += features[j][i];
		}
		mean[i] /= numOfDataPoints;
	}

	return mean;
}

// Calculate the standard deviation for feature scaling
std::vector<float> LogisticRegression::GetStandardDeviation(const std::vector<float>& mean) const
{
	std::vector<float> standardDeviation(numOfFeatures, 0);

	for (int i = 0; i < numOfFeatures; i++) {
		for (int j = 0; j < numOfDataPoints; j++) {
			standardDeviation[i] += (features[j][i] - mean[i]) * (features[j][i] - mean[i]);
		}
		standardDeviation[i] = std::sqrt(standardDeviation[i] / (numOfDataPoints - 1));
	}

	return standardDeviation;
}

// Train the model using Gradient Descent
void LogisticRegression::Train(const float& learningRate, const int maxIterations)
{
	const float convergenceThreshold = 2e-10; // Convergence threshold
	const int logStep = 1000; // Logging step interval

	int i;
	for (i = 1; i <= maxIterations; i++) {
		float prevCost = Loss();
		std::vector<float> prevCoefficients = coefficients;
		float prevIntercept = intercept;

		// Compute gradients
		std::vector<float> coeffSlope = CoeffGradient();
		float interceptSlope = InterceptGradient();

		// Update  parameters
		for (int j = 0; j < numOfFeatures; j++) {
			coefficients[j] -= learningRate * coeffSlope[j];
		}
		intercept -= learningRate * interceptSlope;

		float currentCost = Loss();

		// Check for convergence
		if (i > 1 && (currentCost > prevCost || prevCost - currentCost <= convergenceThreshold)) {
			// Early stoppage due to convergence
			coefficients = prevCoefficients;
			intercept = prevIntercept;
			break;
		}

		if (i % logStep == 0) {
			std::cout << "Iteration #" << i << ", Cost: " << currentCost << '\n';
		}
	}

	// Output final cost to console for learning rate adjustments
	std::cout << "Final cost after " << i - 1 << " iterations : " << Loss() << "\n \n";
}

// Get the model parameters (coefficients and intercept)
std::vector<float> LogisticRegression::GetParameters() const
{
	std::vector<float> parameters(coefficients);
	parameters.push_back(intercept);
	return parameters;
}