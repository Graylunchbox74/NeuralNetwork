#include "neuralNet/neuralNetwork.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <unistd.h>

#define NUM_TRAINING_THREADS 3


float accuracies[NUM_TRAINING_THREADS];
std::vector<bool> isFinished(NUM_TRAINING_THREADS);
int inc = 2;


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

void testAndSave(NeuralNetwork& n, float accuracy){
	std::cout << "Accuracy3: " << accuracy/100 <<std::endl;
	inc++;
	n.SaveNetwork("./SavedNetworks/saved" + std::to_string(inc), accuracy);
	return;
	
	// std::string directory = "./SavedNetworks/saved";
	// float oldAccuracy;
	// std::ifstream inputFile(directory + std::to_string(inc));
	// inputFile >> oldAccuracy;
	// if(accuracy > oldAccuracy){
	// 	inc++;
	// 	n.SaveNetwork("./SavedNetworks/saved" + std::to_string(inc), accuracy);
	// }
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
	accuracies[my_id] = accuracy;
	isFinished[my_id] = true;
	testAndSave(n, accuracy);
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

int main(){

	//init
	std::vector<NeuralNetwork> networks;
	srand(time(NULL));
	for(int i = 0; i < NUM_TRAINING_THREADS; i++){
		isFinished[i] = false;
		networks.push_back(makeRandomNeuralNetwork());
		accuracies[i] = 0;
	}

	std::vector<std::thread> threads;
	std::vector<float> accuracies(NUM_TRAINING_THREADS);
	for(int i = 0; i < NUM_TRAINING_THREADS; i++){
		threads.push_back(std::thread(trainRandomNeuralNetwork,std::ref(networks[i]), i));
	}

	while(1){
		for(int i = 0; i < NUM_TRAINING_THREADS; i++){
			if(isFinished[i])
			{
				threads[i].join();
				accuracies[i] = 0;
				networks[i] = makeRandomNeuralNetwork();
				threads[i] = std::thread(trainRandomNeuralNetwork,std::ref(networks[i]), i);
				isFinished[i] = false;
			}
		}

		usleep(1000000);
	}

	//lets fine tune this network

	return 0;
}
