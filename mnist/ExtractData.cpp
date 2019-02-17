#include <fstream>
#include <vector>
#include <iostream>
#include <string>

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

int main(){
    std::ifstream imagesFile("images");
    std::ifstream labelsFile("labels");

    char dummy[16];

    //get dummy data out of file
	imagesFile.read(dummy, 16);
	labelsFile.read(dummy, 8);

    int label;
    std::vector<float> image;

    for (int i = 0; i < 10000; i++)
    {
        image = getNextImage(imagesFile);
        label = getNextLabel(labelsFile);
        std::ofstream outputFile(std::to_string(i).c_str());
        outputFile << label << "\n";
        for (int x = 0; x < 28; ++x){
            for (int y = 0; y < 28; ++y){
            outputFile << image[x*28 + y] << " ";
           }
            outputFile << "\n";
        }
        outputFile.close();
    }

}