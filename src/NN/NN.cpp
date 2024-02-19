#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>

using namespace std;

// defining all global variables
const int MSE = 0;
const int ACCURACY = 1;
const int LOSS = 2;
const int BCE = 3;
const int LINEAR = 0;
const int RELU = 1;
const int SIGMOID = 2;
const int LEAKYRELU = 3;
const int SOFTMAX = 4;

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

void WriteAccuracy(float* acc, int amount)
{
	ofstream CSV;
	CSV.open("accuracy.csv");
	for (int i = 0; i < amount; i++) {
		if (i != 0) {
			CSV << ",\n";
		}
		stringstream string;
		string << acc[i];
		CSV << string.str();
	}
	CSV.close();
}

vector<vector<float>> Normalise(vector<vector<float>> data) {
	vector<vector<float>> output;
	vector<float> maxes;
	for (int i = 0; i < data[0].size(); i++) {
		maxes.push_back(data[0][i]);
	}
	vector<float> mins;
	for (int i = 0; i < data[0].size(); i++) {
		mins.push_back(data[0][i]);
	}
	// finds maxes and mins
	for (int i = 1; i < data.size(); i++) {
		for (int t = 0; t < data[i].size(); t++) {
			if (maxes[t] < data[i][t]) {
				maxes[t] = data[i][t];
			}
			if (mins[t] > data[i][t]) {
				mins[t] = data[i][t];
			}
		}
	}
	output.push_back(maxes);
	output.push_back(mins);
	// normalises all the data
	for (int i = 0; i < data.size(); i++) {
		vector<float> temp = {};
		for (int t = 0; t < data[i].size(); t++) {
			temp.push_back((data[i][t] - mins[t]) / (maxes[t] - mins[t]));
		}
		output.push_back(temp);
	}
	return output;
}

class Layer {
private:
	float** weights;
	float** weightsup;
	float** lastweightsup;
	float* biases;
	float* biasesup;
	float* lastbiasesup;
	float* inputstore;
	float* resultstore;
	int activation;
	float* layercost;
	float alpha;
	float epsilon;
	float* lastcost;
	float L1;
	float L2;
	float squaredgradsum;
	float rho;
	float decay;
public:
	int input;
	int output;
	float momentum;
	// creates layer for network
	Layer() {
		;
	}
	Layer(int input, int output, int activation = LINEAR, float alpha = 0.001, float epsilon = 1e-8, float rho = 1e-7, float momentum = 0, float L1 = 0, float L2 = 0, float decay = 1e-7) {
		this->inputstore = new float[input];
		this->resultstore = new float[output];
		this->input = input;
		this->output = output;
		this->weights = new float* [input];
		this->biases = new float[output]();
		this->weightsup = new float* [input]();
		this->lastweightsup = new float* [input]();
		this->biasesup = new float[output]();
		this->lastbiasesup = new float[output]();
		this->activation = activation;
		this->layercost = new float[input]();
		this->alpha = alpha;
		this->epsilon = epsilon;
		this->momentum = momentum;
		this->L1 = L1;
		this->L2 = L2;
		this->squaredgradsum = 0;
		this->rho = rho;
		this->decay = decay;
		for (int i = 0; i < input; i++) {
			this->weights[i] = new float[output];
			this->weightsup[i] = new float[output]();
			this->lastweightsup[i] = new float[output]();
			for (int t = 0; t < output; t++) {
				// assigns a random float between -0.5 and 0.5 to each weight
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				r = r - 0.5f;
				this->weights[i][t] = r;
			}
		}
	}
	Layer(int input, int output, float** weights, float* biases, int activation = LINEAR, float alpha = 0.01, float epsilon = 1e-8, float rho = 1e-7, float momentum = 0, float L1 = 0, float L2 = 0, float decay = 1e-7) {
		this->inputstore = new float[input];
		this->resultstore = new float[output];
		this->input = input;
		this->output = output;
		this->weights = new float* [input]();
		this->biases = new float[output]();
		this->weightsup = new float* [input]();
		this->lastweightsup = new float* [input]();
		this->biasesup = new float[output]();
		this->lastbiasesup = new float[output]();
		this->activation = activation;
		this->layercost = new float[input]();
		this->alpha = alpha;
		this->epsilon = epsilon;
		this->momentum = momentum;
		this->L1 = L1;
		this->L2 = L2;
		this->rho = rho;
		this->squaredgradsum = 0;
		this->decay = decay;
		for (int i = 0; i < input; i++) {
			this->weights[i] = new float[output];
			this->weightsup[i] = new float[output]();
			this->lastweightsup[i] = new float[output]();
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
		// applies activation function
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
		case LEAKYRELU:
			for (int i = 0; i < output; i++) {
				if (result[i] < 0) {
					result[i] = 0.01 * result[i];
				}
			}
			break;
		case SIGMOID:
			for (int i = 0; i < output; i++) {
				result[i] = 1 / (1 + exp(-result[i]));
			}
			break;
		case SOFTMAX:
			float expsum = 0;
			for (int i = 0; i < output; i++) {
				expsum += exp(result[i]);
			}
			for (int i = 0; i < output; i++) {
				result[i] = exp(result[i]) / expsum;
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
		case LEAKYRELU:
			for (int i = 0; i < this->output; i++) {
				if (this->resultstore[i] < 0) {
					cost[i] = 0.01 * cost[i];
				}
			}
			break;
		case SIGMOID:
			for (int i = 0; i < this->output; i++) {
				cost[i] = cost[i] * exp(resultstore[i]) / pow((1 + exp(resultstore[i])), 2);
			}
			break;
		case SOFTMAX:
			float expsum = 0;
			for (int i = 0; i < output; i++) {
				expsum += exp(resultstore[i]);
			}
			for (int i = 0; i < output; i++) {
				cost[i] = cost[i] * ((expsum - exp(resultstore[i])) * exp(resultstore[i]) / pow(expsum, 2));
			}
			break;
		}
		this->squaredgradsum = (1 - this->rho) * this->squaredgradsum;
		for (int i = 0; i < output; i++) {
			this->squaredgradsum += this->rho * pow(cost[i], 2);
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
	void Update(int batchsize) {
		float change = 0;
		float step = this->alpha / (this->epsilon + sqrt(this->squaredgradsum));
		for (int i = 0; i < output; i++) {
			change = step * this->biasesup[i] / batchsize + this->momentum * this->lastbiasesup[i];
			if (this->biases[i] > 0) {
				change += L1;
			}
			else {
				change -= L1;
			}
			change += 2 * L2 * this->biases[i];
			this->biases[i] -= change;
			this->lastbiasesup[i] = change;
			this->biasesup[i] = 0;
		}
		for (int i = 0; i < input; i++) {
			for (int t = 0; t < output; t++) {
				change = step * this->weightsup[i][t] / batchsize + this->momentum * this->lastweightsup[i][t];
				if (this->weights[i][t] > 0) {
					change += L1;
				}
				else {
					change -= L1;
				}
				change += 2 * L2 * this->weights[i][t];
				this->weights[i][t] -= change;
				this->lastweightsup[i][t] = change;
				this->weightsup[i][t] = 0;
			}
		}
		this->alpha = this->alpha * (1 - this->decay);
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
	float* lastdcost;
	float* dcost;
	int costfunc;
	int accuracyfunc;
	int out;
	int largestlayersize;
	float momentum;
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
		this->momentum = momentum;
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
		case BCE:
			for (int i = 0; i < this->out; i++) {
				this->cost[i] = -(truevals[i] * log(this->result[i]) + (1 - truevals[i]) * log(1 - this->result[i]));
			}
			break;
			// all bellow not to be used for anything but analyzing the network
		case ACCURACY:
			for (int i = 0; i < this->out; i++) {
				this->cost[i] = ((result[i] - truevals[i]) * (result[i] - truevals[i]) < 0.25) ? 1 : 0;
			}
			break;
		case LOSS:
			for (int i = 0; i < this->out; i++) {
				this->cost[i] = abs(result[i] - truevals[i]);
			}
			break;
		}
		return this->cost;
	}
	float* CalculateAccuracy(float* truevals) {
		// checks which cost function should be used
		switch (this->accuracyfunc) {
		case MSE:
			for (int i = 0; i < this->out; i++) {
				float temp = (truevals[i] - this->result[i]);
				this->cost[i] = temp * temp;
			}
			break;
		case BCE:
			for (int i = 0; i < this->out; i++) {
				this->cost[i] = -(truevals[i] * log(this->result[i]) + (1 - truevals[i]) * log(1 - this->result[i]));
			}
			break;
			// all bellow not to be used for anything but analyzing the network
		case ACCURACY:
			if (this->out == 1) {
				for (int i = 0; i < this->out; i++) {
					this->cost[i] = ((result[i] - truevals[i]) * (result[i] - truevals[i]) < 0.25) ? 1 : 0;
				}
			}
			else {
				int maxind = 0;
				float maxval = result[0];
				int trueind = 0;
				for (int i = 0; i < this->out; i++) {
					if (maxval < result[i]) {
						maxval = result[i];
						maxind = i;
					}
					if (truevals[i] == 1) {
						trueind = i;
					}
				}
				if (trueind == maxind) {
					for (int i = 0; i < this->out; i++) {
						this->cost[i] = 1;
					}
				}
				else {
					for (int i = 0; i < this->out; i++) {
						this->cost[i] = 0;
					}
				}
			}
			break;
		case LOSS:
			for (int i = 0; i < this->out; i++) {
				this->cost[i] = abs(result[i] - truevals[i]);
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
		case BCE:
			for (int i = 0; i < this->out; i++) {
				this->dcost[i] = -((result[i] - truevals[i]) / ((result[i] - 1) * result[i]));
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
	void Update(int batchsize = 1) {
		for (int i = this->layeram - 1; i >= 0; i--) {
			this->layers[i].Update(batchsize);
		}
	}
	float Accuracy(float** inputs, float** trues, int inpsize, int outsize, int inpamount) {
		float sumcost = 0;
		float* input = new float[0];
		float* _true = new float[0];
		for (int i = 0; i < inpamount; i++) {
			input = new float[inpsize];
			for (int t = 0; t < inpsize; t++) {
				input[t] = inputs[i][t];
			}
			float* _true = new float[outsize];
			for (int t = 0; t < outsize; t++) {
				_true[t] = trues[i][t];
			}
			float* out = this->Pass(input);
			float* tempcost = this->CalculateAccuracy(_true);
			if (this->accuracyfunc != ACCURACY) {
				for (int t = 0; t < outsize; t++) {
					sumcost += tempcost[t];
				}
			}
			else {
				for (int t = 0; t < outsize; t++) {
					sumcost += tempcost[t] / outsize;
				}
			}
		}
		return sumcost / inpamount;
	}
	void Train(float** inputs, float** trues, int inpsize, int outsize, int inpamount, int epochs, int minibatch = 1) {
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
			if (epoch % minibatch == 0) {
				this->Update(minibatch);
			}
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