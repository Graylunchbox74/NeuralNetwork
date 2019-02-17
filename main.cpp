#include "neuralNet/neuralNetwork.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>

inline std::vector<float> getNextImage(std::ifstream& imageFile){
	//784
	std::vector<float> image;
	char buff[784];
	imageFile.read(buff, sizeof(buff));
	for (int i = 0; i < 784; ++i)
	{
		//image.push_back(float(c)/255);
		image.push_back((unsigned char)(buff[i]));
	}
	return image;
}

inline int getNextLabel(std::ifstream& labelFile){
	char c;
	labelFile.read(&c,1);
	return int(c);
}


//newcomment
int main(){

	srand(time(NULL));
	//start reading file
	std::ifstream imageFile("mnist/training_data/images");
	std::ifstream labelFile("mnist/training_data/labels");

	char currentInputImageFile, currentInputLabelFile;

	//get through the dumb data
	for (int i = 0; i < 16; i++)
	{
		imageFile.read(&currentInputLabelFile, 1);
	}

	for (int i = 0; i < 8; ++i)
	{
		labelFile.read(&currentInputLabelFile, 1);
	}
	// std::vector<float> image;
	// std::vector<float> expectedOutput = {0,0,0,0,0,0,0,0,0,0};
	// int label;
	// std::vector<int> numNodes = {784,16,10};
	// NeuralNetwork n(numNodes.size(),numNodes,2);

	// std::vector<std::vector<float>> multiBatchImages;
	// std::vector<std::vector<float>> multiBatchLabels;
	// for (int i = 0; i < 50000/25; ++i)
	// {
	// 	for (int x = 0; x < 25; ++x)
	// 	{
	// 		image = getNextImage(imageFile);
	// 		label = getNextLabel(labelFile);
	// 		expectedOutput[label] = 1;
	// 		//n.TrainNetworkSingleInstance(image,expectedOutput);
	// 		multiBatchImages.push_back(image);
	// 		multiBatchLabels.push_back(expectedOutput);

	// 		expectedOutput[label] = 0;
	// 	}
	// 	n.TrainNetworkMultipleInstance(multiBatchImages,multiBatchLabels);
	// 	std::cout <<"COST: "<< n.cost << std::endl;
	// 	//usleep(1000000000);
	// 	multiBatchLabels.clear();
	// 	multiBatchImages.clear();
	// }

	std::vector<int> numNodes = {5,5,5};
	NeuralNetwork n(numNodes.size(), numNodes, 1);
	std::vector<float> input = {0,0,0,0,0};
	std::vector<float> expectedOutput = {0,0,0,0,0};
	for (int i = 0; i < 3; ++i)
	{
		//print out current weights
		std::cout << std::endl << "Starting training instance number: " << i <<std::endl;
		std::cout << "------------------------------------------------------" <<std::endl;
		for (int x = 1; x < 3; ++x)
		{
			std::cout << "	--- weights from " << x-1 << " to " << x << " ---" << std::endl;
			for (int y = 0; y < 5; ++y)
			{
				std::cout << "		neuron number: " << y << " Bias: " << n.layers[x].neurons[y].bias<<std::endl;
				for (int z = 0; z < 5; ++z)
				{
					std::cout << "			" << n.layers[x].neurons[y].weights[z] << std::endl;					
				}
			}
			std::cout << "	--------------------------------------------" <<std::endl;
		}
		int randomInput = rand() % 5;
		std::cout << "	input: " << randomInput << std::endl; 
		input[randomInput] = 1;
		expectedOutput[randomInput] = 1;
		n.TrainNetworkSingleInstance(input,expectedOutput);
		input[randomInput] = 0;
		expectedOutput[randomInput] = 0;
		std::cout << "	Output: " << std::endl;
		for (int y = 0; y < 5; ++y)
		{
			std::cout << "		" << n.layers[n.layers.size()-1].neurons[y].value;
		}
		std::cout << std::endl;
		usleep(1000);
	}






	
	// char j;
	// std::cout << "Start testing?";
	// std::cin >> j;

	// int answer;
	// float largestValue;
	// float accuracy = 0;
	// for (int i = 0; i < 10000; ++i)
	// {
	// 	image = getNextImage(imageFile);
	// 	label = getNextLabel(labelFile);
	// 	n.Activate(image);
	// 	answer = 0;
	// 	largestValue = 0.f;
	// 	std::cout << std::endl << "-----------------------------------------" << std::endl;
	// 	for(int y = 0; y < 10; y++){
	// 		if(n.layers[n.layers.size()-1].neurons[y].value > largestValue){
	// 			answer = y;
	// 			largestValue = n.layers[n.layers.size()-1].neurons[y].value;
	// 		}
	// 		std::cout << n.layers[n.layers.size()-1].neurons[y].value << ", ";
	// 	}
	// 	std::cout << std::endl << "-----------------------------------------" << std::endl;
	// 	answer == label ? std::cout << "Correct - " << largestValue << " -> " << answer << std::endl : std::cout << "Incorrect - " << largestValue << " - Guessed: " << answer << " Expected: "<< label << std::endl;
	// 	answer == label ? accuracy++ : accuracy+=0;
	// 	//usleep(100000);
	// }
	// std::cout << "Accuracy: " << accuracy/100 <<std::endl;

	return 0;
}