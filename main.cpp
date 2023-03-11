/**
 * @file main.cpp
 * @author Keoni Burns
 * @brief
 * @version 0.1
 * @date 2022-11-12
 *
 *
 */

#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cnn.h"

using namespace std;
/**
 * @brief reads the input file and creates a vector of ints
 *
 * @param filename string
 * @return vector<int>
 */
vector<vector<long double>> readInput(string filename) {
    vector<vector<long double>> input;
    vector<long double> cur;
    ifstream file(filename, ios::in);

    if (!file.is_open()) {
        cerr << "Error: cannot open file" << filename << endl;
        exit(1);
    }

    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        string word;
        while (iss >> word) {
            cur.push_back(stoi(word));
        }
        input.push_back(cur);
        cur.clear();
    }
    return input;
}

/**
 * @brief reads in the structure file and creates a vector of the data structure
 *
 * @param filename string
 * @return vector<structureData>
 */
vector<structureData*> readStructure(string filename) {
    vector<structureData*> sInput;
    ifstream file(filename, ios::in);

    if (!file.is_open()) {
        cerr << "Error: cannot open file" << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string word;

        iss >> word;
        int id = stoi(word);

        iss >> word;
        char Ltype = word[0];

        iss >> word;
        int numfilters = stoi(word);

        iss >> word;
        int filtersize = stoi(word);

        iss >> word;
        int stride = stoi(word);

        iss >> word;
        int matrixDimension = stoi(word);

        iss >> word;
        int channels = stoi(word);

        iss >> word;
        int activation = stoi(word);

        iss >> word;
        double bias = stod(word);

        switch (Ltype) {
            case INPUT:
                sInput.push_back(
                    new Input(id, Ltype, numfilters, filtersize, stride, matrixDimension, channels, activation, bias));
                break;
            case CONVOLUTION:
                sInput.push_back(new Convolution(id, Ltype, numfilters, filtersize, stride, matrixDimension, channels,
                                                 activation, bias));
                break;
            case AVERAGE_POOLING:
                sInput.push_back(new AvgPooling(id, Ltype, numfilters, filtersize, stride, matrixDimension, channels,
                                                activation, bias));
                break;
            case MAX_POOLING:
                sInput.push_back(new MaxPooling(id, Ltype, numfilters, filtersize, stride, matrixDimension, channels,
                                                activation, bias));
                break;
            case FULLY_CONNECTED:
                sInput.push_back(new Connected(id, Ltype, numfilters, filtersize, stride, matrixDimension, channels,
                                               activation, bias));
                break;
        }
    }

    return sInput;
}

/**
 * @brief creates a vector of weights from the input file
 *
 * @param filename
 * @return vector<vector<long double>>
 */
vector<vector<long double>> readWeights(string filename) {
    vector<vector<long double>> weights;
    vector<long double> cur;
    vector<string> stringWeights;
    ifstream file(filename, ios::in);

    if (!file.is_open()) {
        cerr << "Error: cannot open file" << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string word;
        while (iss >> word) {
            cur.push_back(stold(word));
        }

        weights.push_back(cur);
        cur.clear();
    }

    return weights;
}

/**
 * @brief driver function
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char** argv) {
    vector<vector<long double>> in = readInput(argv[1]);
    vector<vector<long double>> flatWeights = readWeights(argv[2]);
    vector<structureData*> data = readStructure(argv[3]);
    CNN net;

    net.run(in, flatWeights, data, in.size());
    return 0;
}