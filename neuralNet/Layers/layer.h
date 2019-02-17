#include <vector>
#include <thread>
#include "Neurons/neuron.h"

#define NUM_THREADS 4

class Layer{
public:
	std::vector<Neuron> neurons;
	
	Layer(int numberOfNeurons, int numberOfInputNodes);
	Layer(int numberOfInputNeurons);

	void ActivateLayer(std::vector<float>& previousLayer);
	void ResetLayer();

	std::vector<float> GetLayerValues();
};