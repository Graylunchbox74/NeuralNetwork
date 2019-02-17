#include <vector>
#include <cstdlib>
#include <cmath>

class Neuron{
public:
	std::vector<float> weights;
	float delta;
	float preSigValue;
	float bias;
	float value;

	//initialize neuron with random values for weights and bias
	Neuron(int previousNeuronCount);
	Neuron(int previousNeuronCount, std::vector<float> defaultWeight, float defaultBias);


	//run the algorithm to obtain this neurons value
	void Activate(std::vector<float>& inputValues);

	//reset the NeuronValue for the next test
	void ResetNeuronValue();

private:
	float FastSigmoid(float x);
};

