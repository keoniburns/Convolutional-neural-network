/**
 * @file CNN.h
 * @author Keoni Burns
 * @brief
 * @version 0.1
 * @date 2022-11-12
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CNN_H
#define CNN_H

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief enum used to make parsing the data type easier
 *
 */
enum { INPUT = 'I', CONVOLUTION = 'C', AVERAGE_POOLING = 'A', MAX_POOLING = 'M', FULLY_CONNECTED = 'F' };

/**
 * @brief basic matrix class that has a 3d vector
 *
 */
class Matrix {
   public:
    Matrix();
    Matrix(vector<vector<vector<long double>>> matrix) { mMatrix = matrix; };
    vector<vector<vector<long double>>> getVec() { return mMatrix; }
    void DisplayInput(int precision = 5);
    void clear() { mMatrix.clear(); };

   private:
    vector<vector<vector<long double>>> mMatrix;
};

/**
 * @brief this abstract class is responsible for determining which action we will take on the respective layer
 * additionally it passes all of the relevant information to its children functions
 *
 */
class structureData {
   public:
    structureData(int id, char Ltype, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
                  int activation, double bias);

    /**
     * @brief displays the structure data in a formatted way
     *
     */
    void displayData() {
        cout << "id: " << mid << " | type: " << mtype << " | number of filters: " << mnumFilters
             << " | filter size: " << mfilterSize << " | stride: " << mstride
             << " | matrix dimension: " << mmatrixDimension << " | channels: " << mchannels
             << " | activation: " << mactivation << " | bias " << mbias << endl;
    };

    /**
     * @brief displays weights of the respective datastructure
     *
     */
    void displayW() {
        for (int c = 0; c < mWeights.size(); c++) {
            for (int i = 0; i < mWeights[c].size(); i++) {
                cout << "[ ";
                for (int j = 0; j < mWeights[c][i].size(); j++) {
                    cout << mWeights[c][i][j] << " ";
                }
                cout << " ]" << endl;
            }
        }
    }

    /**
     * @brief virtual matrix to share between the children of structuredata super class
     *
     * @param input
     * @return Matrix
     */
    virtual Matrix doTheThing(Matrix input) {
        cerr << "NOT OVERIDED" << endl;
        return Matrix();
    };

    Matrix activation(Matrix input);
    int getType() { return mtype; };
    int getNumFilters() { return mnumFilters; };
    int getside() { return mmatrixDimension; };

    // this creates the weights and also formats them to be square
    void makeWeights(vector<long double> &a);

    // this creates the weights for the fully connected layer
    void fullConWeights(int numWeights, vector<vector<long double>> &a);

   protected:
    int mid;
    char mtype;
    int mnumFilters;
    int mfilterSize;
    int mstride;
    int mmatrixDimension;
    int mchannels;
    int mactivation;
    double mbias;
    vector<vector<vector<long double>>> mWeights;
};

/**
 * @brief convolution class that deals with convolving
 *
 */
class Convolution : public structureData {
   public:
    Convolution(int id, char Ltype, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
                int activation, double bias);
    /**
     * @brief the convolution helper takes in the channel row col and the input matrix and performs dot multiplication
     * with the given weights
     *
     * @param c
     * @param row
     * @param col
     * @param input
     * @return long double
     */
    long double convHelper(int c, int row, int col, Matrix &input);
    void displayWeights();
    /**
     * @brief iterates the resulting matrix and populates each cell in the the 3d vector
     *
     * @param input
     * @return Matrix
     */
    Matrix doTheThing(Matrix input) override;
};

/**
 * @brief deals with maxpool functionality
 *
 */
class MaxPooling : public structureData {
   public:
    MaxPooling(int id, char Ltype, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
               int activation, double bias);
    /**
     * @brief creates the resulting matrix from the filter and the input vector
     *
     * @param input
     * @return Matrix
     */
    Matrix doTheThing(Matrix input) override;
    /**
     * @brief takes the row col and channel as well as the input vector in order to populate the resulting matrix
     *
     * @param input
     * @param c
     * @param row
     * @param col
     * @return long double
     */
    long double maxHelper(vector<vector<vector<long double>>> &input, int c, int row, int col);
};

/**
 * @brief deals with avg pooling functionality
 *
 */
class AvgPooling : public structureData {
   public:
    AvgPooling(int id, char Ltype, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
               int activation, double bias);
    /**
     * @brief returns the resulting matrix of an average pooling operation
     *
     * @param input
     * @return Matrix
     */
    Matrix doTheThing(Matrix input) override;
    /**
     * @brief populates the resulting matrix with the average of the values within the filter size
     *
     * @param input
     * @param chan
     * @param rows
     * @param col
     * @return long double
     */
    long double avgHelper(vector<vector<vector<long double>>> &input, int chan, int rows, int col);
};

/**
 * @brief creates our input matrix
 *
 */
class Input : public structureData {
   public:
    Input(int id, char Ltype, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
          int activation, double bias);
    Matrix doTheThing(Matrix input) override;
};

/**
 * @brief deals with our fully connected layers
 *
 */
class Connected : public structureData {
   public:
    // constructor
    Connected(int id, char Ltype, int numFilters, int filterSize, int stride, int matrixDimension, int channels,
              int activation, double bias);
    /**
     * @brief populates the fully connected layer output with the respective weights
     *
     * @param count
     * @param chan
     * @param row
     * @param col
     * @param input
     * @return long double
     */
    long double fullHelper(int count, int chan, int row, int col, vector<vector<vector<long double>>> &input);
    /**
     * @brief creates the resulting output matrix
     *
     * @param input
     * @return Matrix
     */
    Matrix doTheThing(Matrix input) override;
};

/**
 * @brief our main cnn class whose main functionality is to run the entire program;
 *
 */
class CNN {
   public:
    CNN();
    // constructor that takes in a structure data
    CNN(vector<structureData *> Data) { mData = Data; };
    // creates our input matrix
    Matrix makeF0(vector<long double> &input);

    // Matrix getInput() { return mInput.getVec(); };
    void run(vector<vector<long double>> &in, vector<vector<long double>> &flatWeights, vector<structureData *> data,
             int iterations);

   private:
    vector<structureData *> mData;
};

#endif