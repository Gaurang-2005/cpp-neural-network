#include <cmath>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>
#include <random>
#include <iostream>

enum class activationFunction {ReLU, sigmoid, Linear};
std::random_device rd;
std::mt19937 rng(rd());

class neuron {
public:
    int nInputs;
    std::vector<float> weights;
    float bias;
    neuron(int prevLayers, activationFunction a) : nInputs{prevLayers}, bias{0.01} {
        if (a == activationFunction::sigmoid || a == activationFunction::Linear) {
            float limit = std::sqrt(3.0f / nInputs);
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (int i = 0; i < nInputs; i++) {
                weights.push_back(dist(rng));
            }
        }
        else {
            float limit = std::sqrt(6.0f / nInputs);
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (int i = 0; i < nInputs; i++) {
                weights.push_back(dist(rng));
            }
        }
        
    }

    std::pair<float, float> activate(activationFunction a, const std::vector<float>& inputs) const {
        float sum{0};

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
    float activationDer(activationFunction a, float sum) {
        switch (a) {
            case activationFunction::Linear: return LinearDer();
            case activationFunction::ReLU: return ReLUDer(sum);
            case activationFunction::sigmoid: return sigmoidDer(sum);
        }
        return 0;
    }
    float Linear(float sum) const {
        return sum;
    }
    float sigmoid(float sum) const {
        return 1/(1 + std::exp(- sum));
    }
    float ReLU(float sum) const {
        if (sum > 0) {
            return sum;
        }
        else return 0;
    }
    float LinearDer() const {
        return 1;
    }
    float ReLUDer(float sum) const {
        if (sum <= 0) return 0;
        else return 1;
    }
    float sigmoidDer(float sum) const {
        float activ = sigmoid(sum);
        return activ * (1 - activ);
    }
    float sumDer(const int& i, const std::vector<float>& inputs) const {

        if (i < int(weights.size())) return inputs[i];
        else if (i == int(weights.size())) return 1;
        else assert(false);
        return 0;
    }
};

class layer {
    int nInputs;
public:
    activationFunction a;
    std::vector<neuron> neuronLayer;
    layer(int n, int in, activationFunction b) : nInputs(in), a(b) {
        for (int i = 0; i < n; i++) {
            neuronLayer.push_back(neuron(in, a));
        }
    }
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& inputs) const {
        assert(nInputs == int(inputs.size()));
        std::pair<float, float> out;
        std::vector<float> first, second;
        for (int i = 0; i < int(neuronLayer.size()); i++) {
            out = neuronLayer[i].activate(a, inputs);
            first.push_back(out.first);
            second.push_back(out.second);
        }
        return {first, second};
    }
};

class neuralNetwork {
public:
    std::vector<layer> layers;
    std::vector<float> predict(const std::vector<float>& inputs) const {
        assert(layers.size() > 0);
        std::vector<float> res(inputs);
        for (int i = 0; i < int(layers.size()); i++) {
            res = (layers[i].forward(res)).second;
        }

        return res;
    }
    std::pair<std::vector<std::vector<float>>,std::vector<std::vector<float>>> forward(const std::vector<float>& inputs) const {
        assert(layers.size() > 0);
        std::pair<std::vector<std::vector<float>>,std::vector<std::vector<float>>> output;
        std::vector<std::vector<float>> first;
        std::vector<std::vector<float>> second;
        std::pair<std::vector<float>, std::vector<float>> temp1 = layers[0].forward(inputs); 
        second.push_back(temp1.second);
        first.push_back(temp1.first);
        for (int i = 1; i < int(layers.size()); i++) {
            std::pair<std::vector<float>, std::vector<float>> temp = layers[i].forward(second.back());
            second.push_back(temp.second);
            first.push_back(temp.first);
        }

        return {first, second};
    }    
    float MSE(std::vector<float> pred, std::vector<float> targ) {
        float err {0};
        assert(pred.size() == targ.size());
        for (int i = 0; i < int(pred.size()); i++) {
            err += (pred[i] - targ[i]) * (pred[i] - targ[i]);
        }

        return err / int(pred.size());
    }
    float MSE(std::vector<std::vector<float>> pred, std::vector<std::vector<float>> targ) {
        float err {0};
        assert(pred.size() == targ.size());
        for (int i = 0; i < int(pred.size()); i++) {
            err += MSE(pred[i], targ[i]);
        }

        return err / int(pred.size());
    }
    float MSE_der(float pred, float targ, int size) {
        return 2.0f / size * (pred - targ); 
    }
    void fit(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& target, const int& epochs, float learningRate) {
        for (int ep = 0; ep < epochs; ep++) {
            std::vector<std::vector<float>> pred;
            for (int i = 0; i < int(inputs.size()); i++) {
                pred.push_back(predict(inputs[i]));
            }
            std::cout<<"dataset MSE at epoch " << ep << ": "<< MSE(pred, target)<<std::endl;
            for (int i = 0; i < int(inputs.size()); i++) {
                std::pair<std::vector<std::vector<float>>,std::vector<std::vector<float>>> out(forward(inputs[i]));
                std::vector<std::vector<float>> sum = out.first;
                std::vector<std::vector<float>> activ = out.second;
                if (target[i].size() != out.second.back().size())
                    throw std::length_error("the size of output of neural network and size of expected values is not same!!");
                std::vector<std::vector<std::vector<float>>> updatedWeights;
                std::vector<std::vector<float>> layerWeightsTemp;
                std::vector<std::vector<float>> delta(activ);
                for (int n2 = int(activ[int(activ.size() - 1)].size()) - 1; n2 >= 0; n2--) {
                    float errorDer = MSE_der(activ[int(activ.size() - 1)][n2], target[i][n2], int(activ[int(activ.size() - 1)].size()));
                    neuron temp(layers[int(activ.size() - 1)].neuronLayer[n2]);
                    float activDer = temp.activationDer(layers[int(activ.size() - 1)].a, sum[int(activ.size() - 1)][n2]);
                    delta[int(activ.size() - 1)][n2] = errorDer * activDer;
                    std::vector<float> updatedWeight;
                    for (int nw = 0; nw <= temp.nInputs; nw++) {
                        float sumDer;
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
                    std::vector<std::vector<float>> layerWeights;
                    for (int n2 = int(activ[n1].size()) - 1; n2 >= 0; n2--) {
                        float tempErrorDer {0};
                        neuron temp(layers[n1].neuronLayer[n2]);
                        for (int k = 0; k < int(sum[n1 + 1].size()); k++) {
                            tempErrorDer += layers[n1 + 1].neuronLayer[k].weights[n2] * delta[n1 + 1][k];
                        }
                        float activDer = temp.activationDer(layers[n1].a, sum[n1][n2]);
                        float errorDer = tempErrorDer;
                        std::vector<float> updatedWeight;
                        delta[n1][n2] = errorDer * activDer;
                        for (int nw = 0; nw <= temp.nInputs; nw++) {
                            float sumDer;
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
};