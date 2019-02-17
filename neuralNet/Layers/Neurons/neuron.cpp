#include "neuron.h"

Neuron::Neuron(int previousNeuronCount){
	for (int i = 0; i < previousNeuronCount; ++i)
	{
		this->weights.push_back(float(rand()%100)/100 - 0.5);
	}
	this->bias = float(rand()%100)/100;
	this->value = 0;
}

Neuron::Neuron(int previousNeuronCount, std::vector<float> defaultWeight, float defaultBias){
	for (int i = 0; i < previousNeuronCount; ++i)
	{
		this->weights.push_back(defaultWeight[i]);
	}
	this->bias = defaultBias;
	this->value = 0;
}

void Neuron::ResetNeuronValue(){
	this->value = 0;
} 

void Neuron::Activate(std::vector<float>& inputValues){
	if (inputValues.size() != this->weights.size())
	{
		exit(-1);
	}
	for (int i = 0; i < inputValues.size(); ++i)
	{
		this->value = this->value + inputValues[i] * this->weights[i];
	}
	this->value += this->bias;
	this->preSigValue = this->value;
	this->value = FastSigmoid(this->value);
}


float Neuron::FastSigmoid(float x){
	return 0.5*(x / (1 + fabs(x))) + 0.5;
}
