#include <fstream>
#include <string>
#include <cstdio>

#define NUMBER_OF_FILES 904

int main(){
    float highestAcc = 0;
    float highestFile = 0;
    float accuracy;
    std::string name = "saved";
    for(int i = 0; i < NUMBER_OF_FILES; i++){
        std::ifstream inputFile(name + std::to_string(i));
        inputFile >> accuracy;
        inputFile.close();
        if(accuracy > highestAcc){
            highestAcc = accuracy;
            highestFile = i;
        }
    }

    for(int i = 0; i < NUMBER_OF_FILES; i++){
        std::ifstream inputFile(name + std::to_string(i));
        inputFile >> accuracy;
        inputFile.close();
        if(i != highestFile){
            remove((name + std::to_string(i)).c_str());
        }
    }
    return 0;
}
