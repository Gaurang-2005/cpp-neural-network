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


