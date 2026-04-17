#include <cstddef>
#include <random>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include "neuralNetwork.h"

std::random_device rd;
std::mt19937 rng(rd());

matrix<double> activate(activationFunction a, const matrix<double>& sum) {

    switch (a) {
        case activationFunction::Linear: return Linear(sum);
        case activationFunction::ReLU: return ReLU(sum);
        case activationFunction::sigmoid: return sigmoid(sum);
    }
    assert(false);
}
matrix<double> activationDer(activationFunction a, const matrix<double>& sum) {
    switch (a) {
        case activationFunction::Linear: return LinearDer(sum);
        case activationFunction::ReLU: return ReLUDer(sum);
        case activationFunction::sigmoid: return sigmoidDer(sum);
    }
    return matrix<double>(0, 0);
}
matrix<double> Linear(const matrix<double>& sum) {
    return sum;
}
matrix<double> sigmoid(const matrix<double>& sum) {
    matrix<double> temp(sum.rows(), 1);
    for (size_t i = 0; i < temp.rows(); i++) temp(i, 0) = 1/(1 + std::exp(- sum(i, 0)));
    return temp;
}
matrix<double> ReLU(const matrix<double>& sum) {
    matrix<double> temp(sum.rows(), 1);
    for (size_t i = 0; i < temp.rows(); i++) (sum(i, 0) > 0)? temp(i, 0) = sum(i, 0):temp(i, 0) = 0;
    return temp;
}
matrix<double> LinearDer(const matrix<double>& sum) {
    return matrix<double>(sum.rows(), 1, 1);
}
matrix<double> ReLUDer(const matrix<double>& sum) {
    matrix<double> temp(sum.rows(), 1);
    for (size_t i = 0; i < temp.rows(); i++) (sum(i, 0) > 0)? temp(i, 0) = 1:temp(i, 0) = 0;
    return temp;
}
matrix<double> sigmoidDer(const matrix<double>& sum) {
    matrix<double> activ = sigmoid(sum);
    matrix<double> temp = (matrix<double>(sum.rows(), 1, 1) - activ);
    return temp.hadamardProduct(activ); 
}

//layer defination
layer::layer(int n, int in, activationFunction b) : nInputs(in), a(b), weights(n, in), bias(n, 1, 0.01) {
    for (int i = 0; i < n; i++) {
        if (a == activationFunction::sigmoid || a == activationFunction::Linear) {
            double limit = std::sqrt(3.0f / nInputs);
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (int j = 0; j < nInputs; j++) {
                weights(i, j) = (dist(rng));
            }
        }
        else {
            double limit = std::sqrt(6.0f / nInputs);
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (int j = 0; j < nInputs; j++) {
                weights(i, j) = (dist(rng));
            }
        }          
    }   
}
std::pair<matrix<double>, matrix<double>> layer::forward(const matrix<double>& inputs) const {
    assert(nInputs == int(inputs.cols()));
    matrix<double> sum = weights * inputs + bias;
    matrix<double> act = activate(a, sum);
    return {sum, act};
}

//neural network
matrix<double> neuralNetwork::predict(const matrix<double>& inputs) const {
    assert(layers.size() > 0);
    matrix<double> res = inputs;
    for (int i = 0; i < int(layers.size()); i++) {
        res = (layers[i].forward(res)).second;
    }

    return res;
}
std::pair<std::vector<matrix<double>>,std::vector<matrix<double>>> neuralNetwork::forward(const matrix<double>& inputs) const {
    assert(layers.size() > 0);
    std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> output;
    std::vector<matrix<double>> first;
    std::vector<matrix<double>> second;
    std::pair<matrix<double>, matrix<double>> temp1 = layers[0].forward(inputs); 
    second.push_back(temp1.second);
    first.push_back(temp1.first);
    for (int i = 1; i < int(layers.size()); i++) {
        temp1 = layers[i].forward(second.back());
        second.push_back(temp1.second);
        first.push_back(temp1.first);
    }

    return {first, second};
}    
double neuralNetwork::MSE(matrix<double> pred, matrix<double> targ) {
    return ((pred - targ).transpose() * (pred - targ))(0, 0) / int(pred.rows());
}
double neuralNetwork::MSE_dataset(matrix<double> pred, matrix<double> targ) {
    return ((pred - targ).transpose() * (pred - targ))(0, 0) / double(pred.cols() * pred.rows());
}
matrix<double> neuralNetwork::MSE_der(matrix<double> pred, matrix<double> targ, int size) {
    return (2.0f / size) * (pred - targ); 
}
void neuralNetwork::fit(const matrix<double>& inputs, const matrix<double>& target, const int& epochs, double learningRate) {
    for (int ep = 0; ep < epochs; ep++) {
        matrix<double> pred = predict(inputs.transpose());
        assert(pred.cols() == target.cols() && pred.rows() == target.rows());
        std::cout<<"dataset MSE at epoch " << ep << ": "<< MSE_dataset(pred, target)<<std::endl;
        for (int i = 0; i < int(inputs.rows()); i++) {
            std::pair<std::vector<matrix<double>>,std::vector<matrix<double>>> out = forward(inputs(i).transpose());
            auto sum = out.first;
            auto activ = out.second;
            if (target(i).cols() != activ.back().rows())
                throw std::length_error("the size of output of neural network and size of expected values is not same!!");
            std::vector<matrix<double>> updatedWeights;
            matrix<double> layerWeightsTemp;
            matrix<double> errorDer = MSE_der(activ[int(activ.size() - 1)], target(i).transpose(), int(activ[int(activ.size() - 1)].rows()));
            matrix<double> activDer = activationDer(layers[int(activ.size() - 1)].a, sum[int(activ.size() - 1)]);
            matrix<double> delta = errorDer.hadamardProduct(activDer);
            matrix<double> sumDer = activ.back().transpose();
            matrix<double> updatedWeight = layers[int(activ.size() - 1)].weights - learningRate * delta * sumDer;
            matrix<double> updatedBias = layers[int(activ.size() - 1)].bias - learningRate * delta;

            updatedWeights.push_back(layerWeightsTemp);
            for (int n1 = int(layers.size()) - 2; n1 >= 0; n1--) {
                std::vector<std::vector<double>> layerWeights;
                for (int n2 = int(activ[n1].size()) - 1; n2 >= 0; n2--) {
                    double tempErrorDer {0};
                    neuron temp(layers[n1].neuronLayer[n2]);
                    for (int k = 0; k < int(sum[n1 + 1].size()); k++) {
                        tempErrorDer += layers[n1 + 1].neuronLayer[k].weights[n2] * delta[n1 + 1][k];
                    }
                    double activDer = temp.activationDer(layers[n1].a, sum[n1][n2]);
                    double errorDer = tempErrorDer;
                    std::vector<double> updatedWeight;
                    delta[n1][n2] = errorDer * activDer;
                    for (int nw = 0; nw <= temp.nInputs; nw++) {
                        double sumDer;
                        if (!(n1 - 1 < 0))
                            sumDer = temp.sumDer(nw, activ[n1 -1]);
                        else {
                            sumDer = temp.sumDer(nw, inputs[i]);
                        }
                        if (nw < temp.nInputs)
                            updatedWeight.push_back(temp.weights[nw] - learningRate * errorDer * activDer * sumDer);
                        else updatedWeight.push_back(temp.bias - learningRate * errorDer * activDer * sumDer);

                    }
                    layerWeights.push_back(updatedWeight);

                }
                updatedWeights.push_back(layerWeights);
            }
            for (int layerI = 0; layerI < int(updatedWeights.size()); layerI++) {
                for (int neuronI = 0; neuronI < int(updatedWeights[layerI].size()); neuronI++) {
                    for (int weightI = 0; weightI < int(updatedWeights[layerI][neuronI].size()); weightI++) {
                        if (weightI < int(layers[int(updatedWeights.size()) - layerI - 1].neuronLayer[int(updatedWeights[layerI].size()) - 1 - neuronI].weights.size())) {
                            layers[int(updatedWeights.size()) - layerI - 1].neuronLayer[int(updatedWeights[layerI].size()) - 1 - neuronI].weights[weightI] = updatedWeights[layerI][neuronI][weightI];
                        }
                        else layers[int(updatedWeights.size()) - layerI - 1].neuronLayer[int(updatedWeights[layerI].size()) - 1 - neuronI].bias = updatedWeights[layerI][neuronI][weightI];
                    }
                }
            }
        }
    }
}
