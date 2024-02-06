#include <iostream>
#include <stdlib.h>

using namespace std;

// defining all global variables
const int MSE = 0;
const int LINEAR = 0;
const int RELU = 1;
const int SIGMOID = 2;

// unoptimized code, fix
float* Dot(float* input, float* weights[], int in, int out) {
	float* result = new float[out]();
	for (int i = 0; i < in; i++) {
		for (int t = 0; t < out; t++) {
			result[t] += weights[i][t] * input[i];
		}
	}
	return result;
}

// transposes the array
float** Transpose(float* arr[], int col, int row) {
	float** result = new float* [col];
	for (int i = 0; i < col; i++) {
		result[i] = new float[row];
		for (int t = 0; t < row; t++) {
			result[i][t] = arr[t][i];
		}
	}
	return result;
}

class Layer {
private:
	float** weights;
	float** weightsup;
	float* biases;
	float* biasesup;
	float* inputstore;
	float* resultstore;
	int activation;
	float* layercost;
	float alpha;
public:
	int input;
	int output;
	// creates layer for network
	Layer() {
		;
	}
	Layer(int input, int output, int activation = LINEAR, float alpha = 0.01) {
		this->inputstore = new float[input];
		this->resultstore = new float[output];
		this->input = input;
		this->output = output;
		this->weights = new float* [input];
		this->biases = new float[output]();
		this->weightsup = new float* [input]();
		this->biasesup = new float[output]();
		this->activation = activation;
		this->layercost = new float[input]();
		this->alpha = alpha;
		for (int i = 0; i < input; i++) {
			this->weights[i] = new float[output];
			this->weightsup[i] = new float[output]();
			for (int t = 0; t < output; t++) {
				// assigns a random float between -0.5 and 0.5 to each weight
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				r = r - 0.5f;
				this->weights[i][t] = r;
			}
		}
	}
	Layer(int input, int output, float** weights, float* biases, int activation = LINEAR, float alpha = 0.01) {
		this->inputstore = new float[input];
		this->resultstore = new float[output];
		this->input = input;
		this->output = output;
		this->weights = new float* [input]();
		this->biases = new float[output]();
		this->weightsup = new float* [input]();
		this->biasesup = new float[output]();
		this->activation = activation;
		this->layercost = new float[input]();
		this->alpha = alpha;
		for (int i = 0; i < input; i++) {
			this->weights[i] = new float[output];
			this->weightsup[i] = new float[output]();
			for (int t = 0; t < output; t++) {
				// assigns set weights for testing
				this->weights[i][t] = weights[i][t];
				this->biases[t] = biases[t];
			}
		}
	}
	float** GetWeights() {
		return this->weights;
	}
	float* GetBiases() {
		return this->biases;
	}
	float* Pass(float* inputs) {
		for (int i = 0; i < this->input; i++) {
			this->inputstore[i] = inputs[i];
		}
		// passes the inputs through the layer with a dot product and then adding the biases
		float* result = Dot(inputs, this->weights, this->input, this->output);
		for (int i = 0; i < output; i++) {
			result[i] += this->biases[i];
			this->resultstore[i] = result[i];
		}
		switch (this->activation) {
		case LINEAR:
			break;
		case RELU:
			for (int i = 0; i < output; i++) {
				if (result[i] < 0) {
					result[i] = 0;
				}
			}
			break;
		case SIGMOID:
			for (int i = 0; i < output; i++) {
				result[i] = 1 / (1 + exp(-result[i]));
			}
			break;
		}
		return result;
	}
	float* PreUpdateCalcs(float* cost) {
		// calculates the derivative of the activation and applies that to the cost
		float* newcost = new float[this->input]();
		switch (this->activation) {
		case LINEAR:
			break;
		case RELU:
			for (int i = 0; i < this->output; i++) {
				if (this->resultstore[i] < 0) {
					cost[i] = 0;
				}
			}
		case SIGMOID:
			for (int i = 0; i < this->output; i++) {
				cost[i] = cost[i] * exp(resultstore[i]) / pow((1 + exp(resultstore[i])), 2);
			}
			break;
		}
		// transposes the weights so that they can be used to calculate the new costs
		float** transposedw = Transpose(this->weights, this->output, this->input);
		// calculates the new costs and the update of the weights
		for (int i = 0; i < this->output; i++) {
			this->biasesup[i] += cost[i];
			for (int t = 0; t < this->input; t++) {
				this->weightsup[t][i] += transposedw[i][t] * cost[i] * this->inputstore[t];
				newcost[t] += this->weightsup[t][i];
			}
			delete[] transposedw[i];
		}
		delete[] transposedw;
		for (int i = 0; i < this->input; i++) {
			this->layercost[i] = newcost[i];
		}
		return newcost;
	}
	void Update() {
		for (int i = 0; i < output; i++) {
			this->biases[i] -= this->alpha * this->biasesup[i];
			this->biasesup[i] = 0;
		}
		for (int i = 0; i < input; i++) {
			for (int t = 0; t < output; t++) {
				this->weights[i][t] -= this->alpha * this->weightsup[i][t];
				this->weightsup[i][t] = 0;
			}
		}
	}
	float** GetWeightChange() {
		return this->weightsup;
	}
	float* GetCost() {
		return this->layercost;
	}
};

class Network {
private:
	Layer* layers;
	int layeram;
	float* result;
	float* cost;
	float* dcost;
	int costfunc;
	int accuracyfunc;
	int out;
	int largestlayersize;
public:
	Network(Layer* layers, int layeram, int costfunc = MSE, int accuracyfunc = MSE) {
		this->layers = layers;
		this->layeram = layeram;
		this->out = layers[layeram - 1].output;
		this->result = new float[this->out];
		this->cost = new float[this->out];
		this->dcost = new float[this->out];
		this->costfunc = costfunc;
		this->accuracyfunc = accuracyfunc;
		this->largestlayersize = 0;
		for (int i = 0; i < this->layeram; i++) {
			if (this->largestlayersize < this->layers[i].input) {
				this->largestlayersize = this->layers[i].input;
			}
			if (this->largestlayersize < this->layers[i].output) {
				this->largestlayersize = this->layers[i].output;
			}
		}
	}
	float* Pass(float* inputs) {
		float* results = new float[this->layers[0].input];
		for (int i = 0; i < this->layers[0].input; i++) {
			results[i] = inputs[i];
		}
		// loops through the layers in the network
		for (int i = 0; i < this->layeram; i++) {
			float* tempresults = this->layers[i].Pass(results);
			delete[] results;
			results = new float[this->layers[i].output]();
			for (int t = 0; t < this->layers[i].output; t++) {
				results[t] = tempresults[t];
			}
			delete[] tempresults;
		}
		for (int i = 0; i < this->out; i++) {
			this->result[i] = results[i];
		}
		return results;
	}
	float* CalculateCost(float* truevals, int costtype = NULL) {
		// checks which cost function should be used
		switch ((costtype != NULL) ? costtype : this->costfunc) {
		case MSE:
			for (int i = 0; i < this->out; i++) {
				float temp = (truevals[i] - this->result[i]);
				this->cost[i] = temp * temp;
			}
			break;
		}
		return this->cost;
	}
	float* CalculateDCost(float* truevals) {
		// checks which function should be used
		switch (costfunc) {
		case MSE:
			for (int i = 0; i < this->out; i++) {
				this->dcost[i] = 2 * (result[i] - truevals[i]);
			}
			break;
		}
		return this->dcost;
	}
	void BackPropagate() {
		float* ccost = new float[this->largestlayersize];
		for (int i = 0; i < this->out; i++) {
			ccost[i] = this->dcost[i];
		}
		// calculated new costs and updates for each layer
		for (int i = this->layeram - 1; i >= 0; i--) {
			float* tempcost = this->layers[i].PreUpdateCalcs(ccost);
			copy(tempcost, tempcost + this->layers[i].input, ccost);
			delete[] tempcost;
		}
		delete[] ccost;
	}
	void Update() {
		for (int i = this->layeram - 1; i >= 0; i--) {
			this->layers[i].Update();
		}
	}
	float Accuracy(float** inputs, float** trues, int inpsize, int outsize, int inpamount) {
		float sumcost = 0;
		float* input = new float[0];
		float* _true = new float[0];
		float* tempcost = new float[0];
		for (int i = 0; i < inpamount; i++) {
			input = new float[inpsize];
			for (int t = 0; t < inpsize; t++) {
				input[t] = inputs[i][t];
			}
			float* _true = new float[outsize];
			for (int t = 0; t < outsize; t++) {
				_true[t] = trues[i][t];
			}
			this->Pass(input);
			tempcost = this->CalculateCost(_true, this->accuracyfunc);
			for (int t = 0; t < outsize; t++) {
				sumcost += tempcost[t];
			}
		}
		return sumcost / inpamount;
	}
	void Train(float** inputs, float** trues, int inpsize, int outsize, int inpamount, int epochs) {
		float* input = new float[inpsize];
		float* _true = new float[outsize];
		for (int epoch = 0; epoch < epochs; epoch++) {
			int r = rand() % inpamount;
			for (int i = 0; i < inpsize; i++) {
				input[i] = inputs[r][i];
			}
			for (int i = 0; i < outsize; i++) {
				_true[i] = trues[r][i];
			}
			float* out = this->Pass(input);
			delete[] out;
			this->CalculateDCost(_true);
			this->BackPropagate();
			this->Update();
		}
	}
	int GetLayerAm() {
		return layeram;
	}
	float** GetLayerWeights(int pos) {
		return this->layers[pos].GetWeights();
	}
	float* GetLayerBiases(int pos) {
		return this->layers[pos].GetBiases();
	}
	float* GetLayerCosts(int pos) {
		return this->layers[pos].GetCost();
	}
};
