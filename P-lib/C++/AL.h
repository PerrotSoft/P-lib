#pragma once
#include <cmath>
#include <vector>
#include <iostream>
struct Neuron {
    double weight = 0.5;  //            
    double bias = 0.5;    //         

    //                   (        )
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    //                   
    double predict(double input) {
        //      *     +         
        double sum = input * weight + bias;
        return sigmoid(sum);
    }

    //                  (                                      )
    void train(double input, double target, double learningRate) {
        double output = predict(input);  //                       
        double error = target - output;  //       

        //                              
        double adjustment = error * output * (1 - output);  //                                

        //                             
        weight += learningRate * adjustment * input;
        bias += learningRate * adjustment;
    }
};
class NeuralNetwork {
private:
    std::vector<Neuron> layer1;  // First layer
    std::vector<Neuron> layer2;  // Second layer

public:
    // Constructor to initialize layers
    NeuralNetwork(int layer1_size, int layer2_size) {
        // Initialize the first layer (e.g., one neuron per input)
        for (int i = 0; i < layer1_size; ++i) {
            layer1.push_back(Neuron());
        }

        // Initialize the second layer
        for (int i = 0; i < layer2_size; ++i) {
            layer2.push_back(Neuron());
        }
    }

    // Run data through the neural network
    std::vector<double> predict(const std::vector<double>& inputs) {
        std::vector<double> layer1_outputs;

        // Pass data through the first layer
        for (size_t i = 0; i < layer1.size(); ++i) {
            layer1_outputs.push_back(layer1[i].predict(inputs[i]));
        }

        std::vector<double> layer2_outputs;
        // Pass data through the second layer
        if (layer1_outputs.size() == layer2.size()) {
            for (size_t i = 0; i < layer2.size(); ++i) {
                layer2_outputs.push_back(layer2[i].predict(layer1_outputs[i]));
            }
        }
        else {
            std::cerr << "Error: The size of the first layer does not match the second layer!" << std::endl;
        }

        return layer2_outputs;
    }
};