#pragma once
#include <vector>
#include "matrix.hpp"

enum class activationFunction {ReLU, sigmoid, Linear};

class neuron {
public:
    int nInputs;
    std::vector<double> weights;
    double bias; 
    
    neuron(int , activationFunction);
    std::pair<double, double> activate(activationFunction , const std::vector<double>&) const;
    double activationDer(activationFunction, double);
    double sumDer(const int& i, const std::vector<double>& inputs) const;
private:
    double Linear(double sum) const;
    double sigmoid(double sum) const;
    double ReLU(double sum) const;
    double LinearDer() const;
    double ReLUDer(double sum) const;
    double sigmoidDer(double) const;
};

class layer {
    int nInputs;
public:
    activationFunction a;
    std::vector<neuron> neuronLayer;

    layer(int, int, activationFunction);
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& inputs) const;
};

class neuralNetwork {
public:
    std::vector<layer> layers;
    std::vector<double> predict(const std::vector<double>& inputs) const;
    std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> forward(const std::vector<double>& inputs) const;
    double MSE(std::vector<double> pred, std::vector<double> targ);
    double MSE(std::vector<std::vector<double>> pred, std::vector<std::vector<double>> targ);
    double MSE_der(double pred, double targ, int size);
    void fit(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& target, const int& epochs, double learningRate);
};