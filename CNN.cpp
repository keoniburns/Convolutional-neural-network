#include "cnn.h"

#include <math.h>

#include <iomanip>
#include <vector>
using namespace std;

CNN::CNN(){};
Matrix::Matrix(){};

structureData::structureData(int id, char type, int numFilters, int filterSize, int stride, int matrixDimension,
                             int channels, int activation, double bias) {
    mid = id;
    mtype = type;
    mnumFilters = numFilters;
    mfilterSize = filterSize;
    mstride = stride;
    mmatrixDimension = matrixDimension;
    mchannels = channels;
    mactivation = activation;
    mbias = bias;
}

Convolution::Convolution(int id, char type, int numFilters, int filterSize, int stride, int matrixDimension,
                         int channels, int activation, double bias)
    : structureData(id, type, numFilters, filterSize, stride, matrixDimension, channels, activation, bias){};

AvgPooling::AvgPooling(int id, char type, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
                       int activation, double bias)
    : structureData(id, type, numFilters, filterSize, stride, matrixDimension, channels, activation, bias){};

MaxPooling::MaxPooling(int id, char type, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
                       int activation, double bias)
    : structureData(id, type, numFilters, filterSize, stride, matrixDimension, channels, activation, bias){};

Input::Input(int id, char type, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
             int activation, double bias)
    : structureData(id, type, numFilters, filterSize, stride, matrixDimension, channels, activation, bias){};

Connected::Connected(int id, char type, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
                     int activation, double bias)
    : structureData(id, type, numFilters, filterSize, stride, matrixDimension, channels, activation, bias){};

void structureData::makeWeights(vector<long double>& a) {
    vector<vector<long double>> weight(mfilterSize, vector<long double>(mfilterSize));

    for (int i = 0; i < mfilterSize; i++) {
        for (int j = 0; j < mfilterSize; j++) {
            weight[i][j] = a[(i * mfilterSize) + j];
        }
    }

    mWeights.push_back(weight);
}

void structureData::fullConWeights(int numWeights, vector<vector<long double>>& a) {
    vector<vector<vector<long double>>> weight(
        mchannels, vector<vector<long double>>(numWeights, vector<long double>(pow(mmatrixDimension, 2))));
    for (int c = 0; c < weight.size(); c++) {
        for (int i = 0; i < weight[c].size(); i++) {
            for (int j = 0; j < weight[c][i].size(); j++) {
                weight[c][i][j] = a[i][j];
            }
        }
    }
    mWeights = weight;
}

/**
 * @brief creates the input layer for the neural network
 *
 * @param input
 * @return Matrix
 */
Matrix Input::doTheThing(Matrix input) { return input; };

long double Convolution::convHelper(int c, int row, int col, Matrix& input) {
    long double dotprod = 0;
    vector<vector<vector<long double>>> inputVec = input.getVec();
    for (int z = 0; z < inputVec.size(); z++) {
        for (int i = row * mstride; i < mfilterSize + (row * mstride); i++) {
            for (int j = col * mstride; j < mfilterSize + (col * mstride); j++) {
                dotprod += (inputVec[z][i][j] * mWeights[c][i - (row * mstride)][j - (col * mstride)]);
            }
        }
    }
    return dotprod;
}

Matrix Convolution::doTheThing(Matrix input) {
    vector<vector<vector<long double>>> inputVec = input.getVec();
    vector<vector<vector<long double>>> dotVectors(
        mnumFilters, vector<vector<long double>>(mmatrixDimension, vector<long double>(mmatrixDimension)));
    // dotvectors is the 3d vector that holds the resulting vectors
    for (int c = 0; c < dotVectors.size(); c++) {                // through channels
        for (int i = 0; i < dotVectors[c].size(); i++) {         // through resulting vectors rows
            for (int j = 0; j < dotVectors[c][i].size(); j++) {  // through the col
                dotVectors[c][i][j] += convHelper(c, i, j, input);
            }
        }
    }
    Matrix output(dotVectors);
    return output;
}

/**
 * helper function to determine bounds
 * incremement by stride
 * create the resulting vec with respect to the formula
 * dothething
 */
long double AvgPooling::avgHelper(vector<vector<vector<long double>>>& input, int chan, int row, int col) {
    long double sum = 0.0;
    for (int i = row * mstride; i < mfilterSize + (row * mstride); i++) {
        for (int j = col * mstride; j < mfilterSize + (col * mstride); j++) {
            // cout << "input channel: " << chan << " | i: " << i << " | j: " << j << endl;
            // cout << "input: " << input[chan][i][j] << endl;
            sum += input[chan][i][j];
        }
    }

    long double result = (sum / (mfilterSize * mfilterSize));
    return result;
}

Matrix AvgPooling::doTheThing(Matrix input) {
    vector<vector<vector<long double>>> inputVec;
    inputVec = input.getVec();
    vector<vector<vector<long double>>> result(
        mchannels, vector<vector<long double>>(mmatrixDimension, vector<long double>(mmatrixDimension)));

    for (int i = 0; i < mchannels; i++) {
        for (int j = 0; j < result[i].size(); j++) {
            for (int z = 0; z < result[i][j].size(); z++) {
                result[i][j][z] = avgHelper(inputVec, i, j, z);
            }
        }
    }

    Matrix output(result);
    return output;
}

long double MaxPooling::maxHelper(vector<vector<vector<long double>>>& input, int c, int row, int col) {
    long double curMax = input[c][row * mstride][col * mstride];

    for (int i = row * mstride; i < mfilterSize + (row * mstride); i++) {
        for (int j = col * mstride; j < mfilterSize + (col * mstride); j++) {
            if (input[c][i][j] > curMax) {
                curMax = input[c][i][j];
            }
        }
    }
    // cout << "current max for channel: " << c << "| row: " << row << "| col: " << col << " is: | " << curMax << endl;
    return curMax;
}

Matrix MaxPooling::doTheThing(Matrix input) {
    vector<vector<vector<long double>>> inputVec;
    inputVec = input.getVec();

    vector<vector<vector<long double>>> result(
        mchannels, vector<vector<long double>>(mmatrixDimension, vector<long double>(mmatrixDimension)));
    for (int i = 0; i < mchannels; i++) {
        for (int j = 0; j < result[i].size(); j++) {
            for (int z = 0; z < result[i][j].size(); z++) {
                result[i][j][z] = maxHelper(inputVec, i, j, z);
            }
        }
    }
    Matrix output(result);
    return output;
}

long double Connected::fullHelper(int count, int chan, int row, int col, vector<vector<vector<long double>>>& input) {
    // cout << "in helper" << endl;
    long double dotprod = 0.0;

    for (int inchan = 0; inchan < input.size(); inchan++) {
        for (int i = 0; i < input[inchan].size(); i++) {
            for (int j = 0; j < input[inchan][i].size(); j++) {
                dotprod += (input[inchan][i][j] *
                            mWeights[chan][j + (i * input[inchan].size())][col + (row * mmatrixDimension)]);
            }
        }
    }
    return dotprod;
}

Matrix Connected::doTheThing(Matrix input) {
    vector<vector<vector<long double>>> inputVec = input.getVec();
    vector<vector<vector<long double>>> result(
        mchannels, vector<vector<long double>>(mmatrixDimension, vector<long double>(mmatrixDimension)));
    int count = 0;
    for (int chan = 0; chan < result.size(); chan++) {
        for (int row = 0; row < result[chan].size(); row++) {
            for (int col = 0; col < result[chan][row].size(); col++) {
                result[chan][row][col] += fullHelper(count, chan, row, col, inputVec);
            }
        }
    }
    Matrix output(result);
    return output;
}

Matrix CNN::makeF0(vector<long double>& input) {
    int size = sqrt(input.size());
    int iterator;
    vector<vector<vector<long double>>> inputVec(1, vector<vector<long double>>(size, vector<long double>(size)));

    for (int i = 0; i < size; i++) {
        iterator = i * size;
        for (int j = 0; j < size; j++) {
            inputVec[0][i][j] = input[j + iterator];
        }
    }

    Matrix inputMatrix(inputVec);
    return inputMatrix;
}

void Matrix::DisplayInput(int precision) {
    cout << std::showpoint << std::fixed << setprecision(precision);
    int count = 0;
    for (auto& channel : mMatrix) {
        // cout << "channel: " << count << endl;
        for (auto& row : channel) {
            for (auto& col : row) {
                cout << col << " ";
            }
        }
        cout << endl;
        count++;
    }
}

void Convolution::displayWeights() {
    int count = 0;
    for (auto& channel : mWeights) {
        cout << "channel: " << count << endl;
        for (auto& row : channel) {
            cout << "[ ";
            for (auto& col : row) {
                cout << col << " ";
            }
            cout << "]" << endl;
        }
    }
}

Matrix structureData::activation(Matrix input) {
    vector<vector<vector<long double>>> result = input.getVec();

    if (mactivation != 0 && mactivation != 1) {
        cerr << "invalid activation type" << endl;
    }

    if (mactivation == 0) {  // sigmoid
        for (int c = 0; c < result.size(); c++) {
            for (int i = 0; i < result[c].size(); i++) {
                for (int j = 0; j < result[c][i].size(); j++) {
                    long double tmp = mbias + result[c][i][j];
                    result[c][i][j] = (1 / (1 + exp(-tmp)));
                }
            }
        }
    } else {  // tanh
        for (int c = 0; c < result.size(); c++) {
            for (int i = 0; i < result[c].size(); i++) {
                for (int j = 0; j < result[c][i].size(); j++) {
                    long double tmp = mbias + result[c][i][j];
                    result[c][i][j] = ((exp(tmp) - exp(-tmp)) / (exp(tmp) + exp(-tmp)));
                }
            }
        }
    }
    Matrix output(result);
    return output;
}

void CNN::run(vector<vector<long double>>& in, vector<vector<long double>>& flatWeights, vector<structureData*> data,
              int iterations) {
    for (int i = 0; i < data.size(); i++) {
        if (data[i]->getType() == CONVOLUTION) {
            for (int j = 0; j < data[i]->getNumFilters(); j++) {
                data[i]->makeWeights(flatWeights[j]);
            }
        } else if (data[i]->getType() == FULLY_CONNECTED) {
            data[i]->fullConWeights(pow(data[i - 1]->getside(), 2), flatWeights);
        }
        flatWeights.erase(flatWeights.begin(), flatWeights.begin() + data[i]->getNumFilters());
    }

    for (int iterations = 0; iterations < in.size(); iterations++) {
        Matrix input(makeF0(in[iterations]));
        // cout << "*************************************************" << endl;
        // cout << "iteration: " << iterations << endl;

        for (int i = 1; i < data.size(); i++) {
            // data[i]->displayData();
            //  cout << "before operation" << endl;
            //  input.DisplayInput(16);

            input = data[i]->doTheThing(input);
            if (data[i]->getType() == MAX_POOLING || data[i]->getType() == AVERAGE_POOLING) {
                // cout << "activation of layer" << i << endl;
                // input.DisplayInput(16);
            } else {
                // cout << "activation of layer" << i << endl;
                input = data[i]->activation(input);
                // input.DisplayInput(16);
            }
        }

        input.DisplayInput(16);
        input.clear();
    }
}