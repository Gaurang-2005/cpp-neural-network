#pragma once
#include <vector>
#include "matrix.hpp"

enum class activationFunction {ReLU, sigmoid, Linear};

matrix<double> activate(activationFunction , const matrix<double>&, const matrix<double>&);
matrix<double> activationDer(activationFunction, double);
matrix<double> sumDer(const int& i, const std::vector<double>& inputs);
matrix<double> Linear(const matrix<double>&);
matrix<double> sigmoid(const matrix<double>&);
matrix<double> ReLU(const matrix<double>&);
matrix<double> LinearDer(const matrix<double>&);
matrix<double> ReLUDer(const matrix<double>&);
matrix<double> sigmoidDer(const matrix<double>&);

class layer {
    int nInputs;
public:
    activationFunction a;
    matrix<double> weights;
    matrix<double> bias;

    layer(int, int, activationFunction);
    std::pair<matrix<double>, matrix<double>> forward(const matrix<double>& inputs) const;
};

class neuralNetwork {
public:
    std::vector<layer> layers;
    matrix<double> predict(const matrix<double>& inputs) const;
    std::pair<std::vector<matrix<double>>,std::vector<matrix<double>>> forward(const matrix<double>& inputs) const;
    double MSE_dataset(matrix<double> pred, matrix<double> targ);
    double MSE(matrix<double> pred, matrix<double> targ);
    matrix<double> MSE_der(matrix<double> pred, matrix<double> targ, int size);
    void fit(const matrix<double>& inputs, const matrix<double>& target, const int& epochs, double learningRate);
};