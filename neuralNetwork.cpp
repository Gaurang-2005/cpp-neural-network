#include <cmath>
#include <cassert>
#include <vector>

enum class activationFunction {ReLU, sigmoid, Linear};

class neuron {
    int nInputs;
    float* weights;
    float bias;
public:
    neuron(int prevLayers) : nInputs{prevLayers}, bias(0.5) {
        weights = new float[nInputs];

        for (int i = 0; i < nInputs; i++) {
            weights[i] = 0.5;
        }

    }

    float activate(activationFunction a, const std::vector<float>& inputs) const {
        float sum{0};

        for (int i = 0; i < nInputs; i++) {
            sum += weights[i] * inputs[i];
        }

        sum += bias;


        switch (a) {
            case activationFunction::Linear: return Linear(sum);
            case activationFunction::ReLU: return ReLU(sum);
            case activationFunction::sigmoid: return sigmoid(sum);
        }
        assert(false);
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

};

class layer {
    std::vector<neuron> neuronLayer;
    int nInputs;
    activationFunction a;
public:
    layer(int n, int in, activationFunction b) : nInputs(in), a(b) {
        for (int i = 0; i < n; i++) {
            neuronLayer.push_back(neuron(in));
        }
    }
    std::vector<float> forward(const std::vector<float>& inputs) const {
        assert(nInputs == inputs.size());
        std::vector<float> out(neuronLayer.size());

        for (int i = 0; i < neuronLayer.size(); i++) {
            out[i] = (neuronLayer[i].activate(a, inputs));
        }

        return out;
    }
};

class neuralNetwork {
public:
    std::vector<layer> layers;
    std::vector<float> forward(const std::vector<float>& inputs) const {
        assert(layers.size() > 0);
        std::vector<float> res(inputs);
        for (int i = 0; i < layers.size(); i++) {
            res = (layers[i].forward(res));
        }

        return res;
    }
};

#include <iostream>
#include <random>

int main() {
    std::cout << "Creating heavy neural network...\n";

    neuralNetwork net;

    int inputSize = 512;

    // Build heavy network
    net.layers.push_back(layer(1024, inputSize, activationFunction::ReLU));
    net.layers.push_back(layer(1024, 1024, activationFunction::ReLU));
    net.layers.push_back(layer(512, 1024, activationFunction::ReLU));
    net.layers.push_back(layer(256, 512, activationFunction::ReLU));
    net.layers.push_back(layer(128, 256, activationFunction::ReLU));
    net.layers.push_back(layer(64, 128, activationFunction::ReLU));
    net.layers.push_back(layer(10, 64, activationFunction::Linear));

    std::cout << "Network created.\n";

    // Create random input
    std::vector<float> input(inputSize);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < inputSize; i++)
        input[i] = dist(rng);

    std::cout << "Running forward pass...\n";

    std::vector<float> output = net.forward(input);

    std::cout << "Output size: " << output.size() << "\n";
    std::cout << "First few outputs:\n";

    for (int i = 0; i < std::min(10, (int)output.size()); i++)
        std::cout << output[i] << " ";

    std::cout << "\nDone.\n";

    return 0;
}
