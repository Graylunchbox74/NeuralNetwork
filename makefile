a.out: main.cpp
	g++ main.cpp neuralNet/neuralNetwork.cpp neuralNet/Layers/layer.cpp neuralNet/Layers/Neurons/neuron.cpp -std=c++11 -O2 -lpthread

all: a.out