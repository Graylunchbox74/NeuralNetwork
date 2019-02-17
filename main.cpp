#include "neuralNet/neuralNetwork.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <unistd.h>

#define NUM_THREADS 4
#define MIN_LAYERS 3
#define MAX_LAYERS 10
#define MIN_NEURONS 10
#define MAX_NEURONS 200
#define MIN_LEARNING_RATE 1
#define MAX_LEARNING_RATE 10
#define TRAINING_ITERATIONS 5000

std::vector<float> accuracies(NUM_THREADS);
std::vector<bool> isFinished(NUM_THREADS);
int inc = 0;


inline std::vector<float> getNextImage(std::ifstream& imageFile){
	//784
	std::vector<float> image;
	int buff;
	for (int i = 0; i < 784; ++i)
	{
		imageFile >> buff;
		//image.push_back(float(c)/255);
		image.push_back(float(buff) / 255);
	}
	return image;
}

inline int getNextLabel(std::ifstream& labelFile){
	int c;
	labelFile >> c;
	return int(c);
}


float trainRandomNeuralNetwork(NeuralNetwork& n, int my_id){
	int label;
	std::vector<float> image;
	std::vector<float> expectedOutput = {0,0,0,0,0,0,0,0,0,0};
	std::string directory = "./mnist/training_data/";
	std::cout <<"threadid: "<< my_id << " Layers: " << n.layers.size() <<std::endl;

	std::vector<std::vector<float>> multiBatchImages;
	std::vector<std::vector<float>> multiBatchLabels;
	for (int i = 0; i < 5000; ++i)
	{
		for (int x = 0; x < 25; ++x)
		{
			std::ifstream inputFile(directory + std::to_string(rand()%50000));
			label = getNextLabel(inputFile);
			image = getNextImage(inputFile);

			expectedOutput[label] = 1;
			//n.TrainNetworkSingleInstance(image,expectedOutput);
			multiBatchImages.push_back(image);
			multiBatchLabels.push_back(expectedOutput);

			expectedOutput[label] = 0;
		}
		n.TrainNetworkMultipleInstance(multiBatchImages,multiBatchLabels);
		//std::cout <<"COST: "<< n.cost << std::endl;
		//usleep(10000);
		multiBatchLabels.clear();
		multiBatchImages.clear();

		if(i % 10 == 0){
			n.learningRate = n.learningRate * 0.999f;
		}
	}


	directory = "./mnist/testing_data/";
	int answer;
	float largestValue;
	float accuracy = 0;
	srand(10);
	for (int i = 0; i < 10000; ++i)
	{
		std::ifstream inputFile(directory + (std::to_string(rand()%10000)));
		label = getNextLabel(inputFile);
		image = getNextImage(inputFile);
		n.Activate(image);
		answer = 0;
		largestValue = 0.f;
		// std::cout << std::endl << "-----------------------------------------" << std::endl;
		for(int y = 0; y < 10; y++){
			if(n.layers[n.layers.size()-1].neurons[y].value > largestValue){
				answer = y;
				largestValue = n.layers[n.layers.size()-1].neurons[y].value;
			}
			// std::cout << n.layers[n.layers.size()-1].neurons[y].value << ", ";
		}
		// std::cout << std::endl << "-----------------------------------------" << std::endl;
		//answer == label ? std::cout << "Correct - " << largestValue << " -> " << answer << std::endl : std::cout << "Incorrect - " << largestValue << " - Guessed: " << answer << " Expected: "<< label << std::endl;
		answer == label ? accuracy++ : accuracy+=0;
		//usleep(100000);
	}
	std::cout << "Accuracy: " << accuracy/100 <<std::endl;
	accuracies[my_id] = accuracy;
	isFinished[my_id] = true;
	return accuracy;
}

NeuralNetwork makeRandomNeuralNetwork(){
	std::vector<int> numNodes;
	int numLayers = rand()%3+3;
	for (int i = 0; i < numLayers; i++)
	{
		if(i == 0){
			numNodes.push_back(784);
		}
		else if(i == numLayers-1)
	{
		numNodes.push_back(10);
		}
		else
		{
			numNodes.push_back(rand()%200+10);
		}	
	}
	float learningRate = float(rand()%100+20)/10;
	NeuralNetwork n(numNodes.size(),numNodes,learningRate);
	return n;
}

void testAndSave(NeuralNetwork& n, float accuracy){
	std::string directory = "./SavedNetworks/saved/";
	float oldAccuracy;
	std::ifstream inputFile(directory + std::to_string(inc));
	inputFile >> oldAccuracy;
	if(accuracy > oldAccuracy){
		inc++;
		n.SaveNetwork("./SavedNetworks/saved" + std::to_string(inc), accuracy);
	}
}

int main(){

	//init
	std::vector<NeuralNetwork> networks;
	srand(time(NULL));
	for(int i = 0; i < NUM_THREADS; i++){
		isFinished[i] = false;
		networks.push_back(makeRandomNeuralNetwork());
		accuracies[i] = 0;
	}

	std::vector<std::thread> threads;
	std::vector<float> accuracies(NUM_THREADS);
	for(int i = 0; i < NUM_THREADS; i++){
		threads.push_back(std::thread(trainRandomNeuralNetwork,std::ref(networks[i]), i));
	}

	while(1){
		for(int i = 0; i < NUM_THREADS; i++){
			if(isFinished[i])
			{
				threads[i].join();
				testAndSave(networks[i],accuracies[i]);
				networks[i] = makeRandomNeuralNetwork();
				threads[i] = std::thread(trainRandomNeuralNetwork,std::ref(networks[i]), i);
				isFinished[i] = false;
			}
		}

		usleep(10000);
	}
	return 0;
}