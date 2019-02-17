#include "neuralNetwork.h"

NeuralNetwork::NeuralNetwork(int numberOfLayers, std::vector<int> numberOfNeuronsPerLayer, int learningRate){
	for (int i = 0; i < numberOfLayers; ++i)
	{
		if (i == 0)
		{
			this->layers.push_back(Layer(numberOfNeuronsPerLayer[i]));
		}
		else
		{
			this->layers.push_back(Layer(numberOfNeuronsPerLayer[i], numberOfNeuronsPerLayer[i-1]));
		}
	}
	this->learningRate = learningRate;
}

void NeuralNetwork::Activate(std::vector<float> input){
	ResetValues();
	for (int i = 0; i < this->layers[0].neurons.size(); ++i)
	{
		this->layers[0].neurons[i].value = input[i];
	}
	for (int i = 1; i < this->layers.size(); ++i)
	{
		std::vector<float> lastLayerValues = this->layers[i-1].GetLayerValues();
		this->layers[i].ActivateLayer(lastLayerValues);
	}
}

void NeuralNetwork::ResetValues(){
	for (int i = 0; i < this->layers.size(); ++i)
	{
		this->layers[i].ResetLayer();
	}
}

void NeuralNetwork::TrainNetworkSingleInstance(std::vector<float> input, std::vector<float>& expectedOutput){
	Activate(input);
	this->cost = 0;
	for (int i = 0; i < this->layers[this->layers.size()-1].neurons.size(); ++i)
	{
		this->cost += (this->layers[this->layers.size()-1].neurons[i].value - expectedOutput[i])*(this->layers[this->layers.size()-1].neurons[i].value - expectedOutput[i]);
	}
	//this->cost /= expectedOutput.size();
	BackPropagateDelta(expectedOutput);
	auto weightChanges = BackPropagateWeight();
	auto biasChanges = BackPropagateBias();
	for (int i = 1; i < layers.size(); ++i)
	{
		for (int x = 0; x < layers[i].neurons.size(); ++x)
		{
			for (int y = 0; y < layers[i].neurons[x].weights.size(); ++y)
			{
				layers[i].neurons[x].weights[y] += weightChanges[i][x][y];
			}
			layers[i].neurons[x].bias += biasChanges[i][x];
		}
	}
}

void NeuralNetwork::TrainNetworkMultipleInstance(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& expectedOutput){
	std::vector<std::vector<std::vector<float>>> changeWeights;
	std::vector<std::vector<std::vector<float>>> tmpWeights;
	std::vector<std::vector<float>> changeBias;
	std::vector<std::vector<float>> tmpBias;
	this->cost = 0;

	for (int i = 0; i < input.size(); ++i)
	{
		Activate(input[i]);
		for (int p = 0; p < this->layers[this->layers.size()-1].neurons.size(); ++p)
		{
			this->cost += (this->layers[this->layers.size()-1].neurons[p].value - expectedOutput[i][p])*(this->layers[this->layers.size()-1].neurons[p].value - expectedOutput[i][p]);
		}

		BackPropagateDelta(expectedOutput[i]);
		if (i == 0)
		{
			changeWeights = BackPropagateWeight();
			changeBias = BackPropagateBias();
		}
		else
		{
			tmpWeights = BackPropagateWeight();
			tmpBias = BackPropagateBias();
			for (int x = 1; x < changeWeights.size(); ++x)
			{
				for (int y = 0; y < changeWeights[x].size(); ++y)
				{
					for (int z = 0; z < changeWeights[x][y].size(); ++z)
					{
						changeWeights[x][y][z] += tmpWeights[x][y][z];
					}
					changeBias[x][y] += tmpBias[x][y];
				}
			}
		}
	}
	for (int x = 1; x < changeWeights.size(); ++x)
	{
		for (int y = 0; y < changeWeights[x].size(); ++y)
		{
			for (int z = 0; z < changeWeights[x][y].size(); ++z)
			{
				layers[x].neurons[y].weights[z] += changeWeights[x][y][z] / expectedOutput.size();
			}
			layers[x].neurons[y].bias += changeBias[x][y] / expectedOutput.size();
		}
	}
	this->cost /= input.size();
	this->cost /= layers[layers.size()-1].neurons.size();
	//this->cost = sqrt(this->cost);
}

inline float NeuralNetwork::FastSigmoidDerivative(float x){
	return 0.5 / ((1.f + fabs(x)) * (1.f + fabs(x)));
}

inline void NeuralNetwork::BackPropagateDelta(std::vector<float>& expectedOutput){
	//change the weights
	for (int i = layers.size()-1; i > 0; --i)
	{
		for (int x = 0; x < layers[i].neurons.size(); ++x)
		{
			if (i != layers.size()-1)
			{
				layers[i].neurons[x].delta = 0;
				for (int q = 0; q < layers[i+1].neurons.size(); ++q)
				{
					layers[i].neurons[x].delta += layers[i+1].neurons[q].delta * FastSigmoidDerivative(layers[i+1].neurons[q].preSigValue) * layers[i+1].neurons[q].weights[x];
				}
			}
			else
			{
				layers[i].neurons[x].delta = 2*(layers[i].neurons[x].value - expectedOutput[x]);
			}
		}
	}
}

std::vector<std::vector<std::vector<float>>> NeuralNetwork::BackPropagateWeight(){
	std::vector<std::vector<std::vector<float>>> changeWeights(this->layers.size());
	//push each neuron to its correct location
	for (int i = 0; i < this->layers.size(); ++i)
	{
		if (i == 0)
		{
			std::vector<std::vector<float>> tmp;
			changeWeights[i] = (tmp);
		}
		else
		{
			changeWeights[i] = (std::vector<std::vector<float>>(this->layers[i].neurons.size()));
			for (int x = 0; x < this->layers[i].neurons.size(); ++x)
			{
				changeWeights[i][x] = (std::vector<float>(this->layers[i-1].neurons.size()));
			}
		}
	}
	for (int i = layers.size()-1; i > 0; --i)
	{
		for (int x = 0; x < layers[i].neurons.size(); ++x)
		{
			for (int y = 0; y < layers[i].neurons[x].weights.size(); ++y)
			{
				//changeWeights[i][x][y] = -1*(learningRate*(layers[i].neurons[x].delta * FastSigmoidDerivative(layers[i].neurons[x].preSigValue) * layers[i-1].neurons[y].value));
				changeWeights[i][x][y] = -1*(learningRate*(layers[i].neurons[x].delta * FastSigmoidDerivative(layers[i].neurons[x].preSigValue) * layers[i-1].neurons[y].value));
			}
		}
	}
	return changeWeights;
}
std::vector<std::vector<float>> NeuralNetwork::BackPropagateBias(){
	std::vector<std::vector<float>> changeBias(this->layers.size());
	for (int i = 0; i < this->layers.size(); ++i)
	{
		if (i == 0)
		{
			changeBias[i] = (std::vector<float>(1));
		}
		else
		{
			changeBias[i] = ((std::vector<float>(layers[i].neurons.size())));
		}
	}
	for (int i = layers.size()-1; i > 0; --i)
	{
		for (int x = 0; x < layers[i].neurons.size(); ++x)
		{
			changeBias[i][x] = -1*learningRate*(layers[i].neurons[x].delta * FastSigmoidDerivative(layers[i].neurons[x].preSigValue));
//			layers[i].neurons[x].bias += -1*learningRate*(layers[i].neurons[x].delta * FastSigmoidDerivative(layers[i].neurons[x].preSigValue));
		}
	}
	return changeBias;
}