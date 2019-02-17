#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "Layers/layer.h"

class NeuralNetwork{
public:
	std::vector<Layer> layers;
	float learningRate;
	float cost;

	NeuralNetwork(int numberOfLayers, std::vector<int> numberOfNeuronsPerLayer, int learningRate);

	//load a network from a saved file
	NeuralNetwork(std::string loadFile);

	//actually run our network with given data
	void Activate(std::vector<float> input);

	//training
	void TrainNetworkMultipleInstance(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& expectedOutput);
	void TrainNetworkSingleInstance(std::vector<float> input, std::vector<float>& expectedOutput);

	//safekeeping
	void SaveNetwork(std::string fileName);

private:
	float FastSigmoidDerivative(float x);
	void BackPropagateDelta(std::vector<float>& expectedOutput);
	std::vector<std::vector<std::vector<float>>> BackPropagateWeight();
	std::vector<std::vector<float>> BackPropagateBias();

	void ResetValues();
};