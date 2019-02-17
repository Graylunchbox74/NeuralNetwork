#include "layer.h"

static void ActivateNeurons(int startId, int numToDo, std::vector<float>& previousLayer, std::vector<Neuron>& neurons){
	for (int i = startId; i < startId+numToDo; ++i)
	{
		neurons[i].Activate(previousLayer);
	}
}

Layer::Layer(int numberOfNeurons, int numberOfInputNodes){
	for (int i = 0; i < numberOfNeurons; ++i)
	{
		Neuron newNeuron(numberOfInputNodes);
		neurons.push_back(newNeuron);
	}
}

Layer::Layer(int numberOfInputNeurons){
	std::vector<float> weights;
	for (int i = 0; i < numberOfInputNeurons; ++i)
	{
		weights.push_back(0.f);
	}

	for (int i = 0; i < numberOfInputNeurons; ++i)
	{
		weights[i] = 1.f;
		Neuron newNeuron(numberOfInputNeurons,weights,0.f);
		neurons.push_back(newNeuron);
		weights[i] = 0.f;
	}
}


void Layer::ActivateLayer(std::vector<float>& previousLayer){
	std::vector<std::thread> threads;
	int numToDo = this->neurons.size() / NUM_THREADS;
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		if (i == NUM_THREADS-1)
		{
			int leftovers = numToDo + (this->neurons.size() % NUM_THREADS);
			threads.push_back(std::thread(ActivateNeurons,numToDo*i,leftovers, std::ref(previousLayer), std::ref(this->neurons)));
		}
		else
		{
			threads.push_back(std::thread(ActivateNeurons,numToDo*i,numToDo, std::ref(previousLayer), std::ref(this->neurons)));
		}
	}
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		threads[i].join();
	}
}

void Layer::ResetLayer(){
	for (int i = 0; i < neurons.size(); ++i)
	{
		neurons[i].ResetNeuronValue();
	}
}

std::vector<float> Layer::GetLayerValues(){
	std::vector<float> values;
	for (int i = 0; i < this->neurons.size(); ++i)
	{
		values.push_back(neurons[i].value);
	}
	return values;
}