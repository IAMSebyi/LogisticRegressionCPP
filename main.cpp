#include "LogisticRegression.h"
#include <fstream>

int main() {
	// Open the input and output files
	std::ifstream inputData("data.txt");
	std::ifstream inputTest("test.txt");
	std::ofstream outputParameters("parameters.txt");

	int numOfFeatures, numOfDataPoints, maxIterations = -1;
	float learningRate, regularizationParam = 0;

	// Read number of features and data points
	inputData >> numOfFeatures >> numOfDataPoints;

	// Initialize vectors for features and targets
	std::vector<std::vector<float>> features(numOfDataPoints, std::vector<float>(numOfFeatures)); // Input independent variables
	std::vector<bool> targets(numOfDataPoints); // Output dependent variables

	// Read features and targets from input file
	for (int i = 0; i < numOfDataPoints; i++) {
		for (int j = 0; j < numOfFeatures; j++) {
			inputData >> features[i][j];
		}
		bool temp;
		inputData >> temp;
		targets[i] = temp;
	}

	// Read learning rate and max iterations
	inputData >> learningRate >> maxIterations >> regularizationParam;

	// Close data input file stream
	inputData.close();

	// Create LogisticRegression model
	LogisticRegression model(features, targets, numOfFeatures, numOfDataPoints, regularizationParam);

	// Train the model
	if (maxIterations == -1) model.Train(learningRate);
	else model.Train(learningRate, maxIterations);

	// Get the model parameters (coefficients and intercept)
	std::vector<float> parameters;
	parameters = model.GetParameters();

	// Write the model parameters to output file
	for (const float& param : parameters) {
		outputParameters << param << " ";
	}

	// Close parameters output file stream
	outputParameters.close();

	// Check if test data points are provided
	int numOfTestDataPoints = -1;
	float threshold = 0.5;
	inputTest >> numOfTestDataPoints >> threshold;

	// Predict and print results for test data points
	if (numOfTestDataPoints != -1) {
		std::vector<float> testFeature(numOfFeatures);

		for (int i = 0; i < numOfTestDataPoints; i++) {
			std::cout << "Result for input { ";

			for (int j = 0; j < numOfFeatures; j++) {
				inputTest >> testFeature[j];
				std::cout << testFeature[j] << " ";
			}

			float probability = model.Predict(testFeature, true);
			if (probability >= threshold) std::cout << "}: TRUE\n";
			else std::cout << "}: FALSE\n";
		}
	}

	// Close test input file stream
	inputTest.close();

	return 0;
}