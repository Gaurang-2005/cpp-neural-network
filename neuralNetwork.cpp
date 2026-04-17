#include <random>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <utility>

#include "neuralNetwork.h"

std::random_device rd;
std::mt19937 rng(rd());

//neuron defination
neuron::neuron(int prevLayers, activationFunction a) : nInputs{prevLayers}, bias{0.01} {
    if (a == activationFunction::sigmoid || a == activationFunction::Linear) {
        double limit = std::sqrt(3.0f / nInputs);
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (int i = 0; i < nInputs; i++) {
            weights.push_back(dist(rng));
        }
    }
    else {
        double limit = std::sqrt(6.0f / nInputs);
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (int i = 0; i < nInputs; i++) {
            weights.push_back(dist(rng));
        }
    }
    
}

std::pair<double, double> neuron::activate(activationFunction a, const std::vector<double>& inputs) const {
    double sum{0};

    for (int i = 0; i < nInputs; i++) {
        sum += weights[i] * inputs[i];
    }

    sum += bias;


    switch (a) {
        case activationFunction::Linear: return {sum, Linear(sum)};
        case activationFunction::ReLU: return {sum, ReLU(sum)};
        case activationFunction::sigmoid: return {sum, sigmoid(sum)};
    }
    assert(false);
}
double neuron::activationDer(activationFunction a, double sum) {
    switch (a) {
        case activationFunction::Linear: return LinearDer();
        case activationFunction::ReLU: return ReLUDer(sum);
        case activationFunction::sigmoid: return sigmoidDer(sum);
    }
    return 0;
}
double neuron::Linear(double sum) const {
    return sum;
}
double neuron::sigmoid(double sum) const {
    return 1/(1 + std::exp(- sum));
}
double neuron::ReLU(double sum) const {
    if (sum > 0) {
        return sum;
    }
    else return 0;
}
double neuron::LinearDer() const {
    return 1;
}
double neuron::ReLUDer(double sum) const {
    if (sum <= 0) return 0;
    else return 1;
}
double neuron::sigmoidDer(double sum) const {
    double activ = sigmoid(sum);
    return activ * (1 - activ);
}
double neuron::sumDer(const int& i, const std::vector<double>& inputs) const {

    if (i < int(weights.size())) return inputs[i];
    else if (i == int(weights.size())) return 1;
    else assert(false);
    return 0;
}
//layer defination
layer::layer(int n, int in, activationFunction b) : nInputs(in), a(b) {
    for (int i = 0; i < n; i++) {
        neuronLayer.push_back(neuron(in, a));
    }
}
std::pair<std::vector<double>, std::vector<double>> layer::forward(const std::vector<double>& inputs) const {
    assert(nInputs == int(inputs.size()));
    std::pair<double, double> out;
    std::vector<double> first, second;
    for (int i = 0; i < int(neuronLayer.size()); i++) {
        out = neuronLayer[i].activate(a, inputs);
        first.push_back(out.first);
        second.push_back(out.second);
    }
    return {first, second};
}

//neural network
std::vector<double> neuralNetwork::predict(const std::vector<double>& inputs) const {
    assert(layers.size() > 0);
    std::vector<double> res(inputs);
    for (int i = 0; i < int(layers.size()); i++) {
        res = (layers[i].forward(res)).second;
    }

    return res;
}
std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> neuralNetwork::forward(const std::vector<double>& inputs) const {
    assert(layers.size() > 0);
    std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> output;
    std::vector<std::vector<double>> first;
    std::vector<std::vector<double>> second;
    std::pair<std::vector<double>, std::vector<double>> temp1 = layers[0].forward(inputs); 
    second.push_back(temp1.second);
    first.push_back(temp1.first);
    for (int i = 1; i < int(layers.size()); i++) {
        std::pair<std::vector<double>, std::vector<double>> temp = layers[i].forward(second.back());
        second.push_back(temp.second);
        first.push_back(temp.first);
    }

    return {first, second};
}    
double neuralNetwork::MSE(std::vector<double> pred, std::vector<double> targ) {
    double err {0};
    assert(pred.size() == targ.size());
    for (int i = 0; i < int(pred.size()); i++) {
        err += (pred[i] - targ[i]) * (pred[i] - targ[i]);
    }

    return err / int(pred.size());
}
double neuralNetwork::MSE(std::vector<std::vector<double>> pred, std::vector<std::vector<double>> targ) {
    double err {0};
    assert(pred.size() == targ.size());
    for (int i = 0; i < int(pred.size()); i++) {
        err += MSE(pred[i], targ[i]);
    }

    return err / int(pred.size());
}
double neuralNetwork::MSE_der(double pred, double targ, int size) {
    return 2.0f / size * (pred - targ); 
}
void neuralNetwork::fit(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& target, const int& epochs, double learningRate) {
    for (int ep = 0; ep < epochs; ep++) {
        std::vector<std::vector<double>> pred;
        for (int i = 0; i < int(inputs.size()); i++) {
            pred.push_back(predict(inputs[i]));
        }
        std::cout<<"dataset MSE at epoch " << ep << ": "<< MSE(pred, target)<<std::endl;
        for (int i = 0; i < int(inputs.size()); i++) {
            std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> out(forward(inputs[i]));
            std::vector<std::vector<double>> sum = out.first;
            std::vector<std::vector<double>> activ = out.second;
            if (target[i].size() != out.second.back().size())
                throw std::length_error("the size of output of neural network and size of expected values is not same!!");
            std::vector<std::vector<std::vector<double>>> updatedWeights;
            std::vector<std::vector<double>> layerWeightsTemp;
            std::vector<std::vector<double>> delta(activ);
            for (int n2 = int(activ[int(activ.size() - 1)].size()) - 1; n2 >= 0; n2--) {
                double errorDer = MSE_der(activ[int(activ.size() - 1)][n2], target[i][n2], int(activ[int(activ.size() - 1)].size()));
                neuron temp(layers[int(activ.size() - 1)].neuronLayer[n2]);
                double activDer = temp.activationDer(layers[int(activ.size() - 1)].a, sum[int(activ.size() - 1)][n2]);
                delta[int(activ.size() - 1)][n2] = errorDer * activDer;
                std::vector<double> updatedWeight;
                for (int nw = 0; nw <= temp.nInputs; nw++) {
                    double sumDer;
                    if (!(int(activ.size() - 2) < 0))
                        sumDer = temp.sumDer(nw, activ[int(activ.size() - 2)]);
                    else {
                        sumDer = temp.sumDer(nw, inputs[i]);
                    }
                    if (nw < temp.nInputs)
                        updatedWeight.push_back(temp.weights[nw] - learningRate * errorDer * activDer * sumDer);
                    else updatedWeight.push_back(temp.bias - learningRate * errorDer * activDer * sumDer);

                }
                layerWeightsTemp.push_back(updatedWeight);
            }
            updatedWeights.push_back(layerWeightsTemp);
            for (int n1 = int(activ.size()) - 2; n1 >= 0; n1--) {
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
