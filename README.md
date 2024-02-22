# NN
## What is it
This is a basic Neural network (which is heavily unoptimised)
This was just a one-month project of mine that I did to get back into the gist of things (I hadn't coded a neural network for about 1 year I think)

## Plans for NN
I do plan on GPU optimising this shitbox, and maybe implementing convolution and memory stuff but this will come at some other point as I currently have another project I'm doing ):

## Example
Apologies for the horrific code this is not pretty
```cpp
int main() {
  int dataam = 0;
  string text;
  ifstream CSV("result.csv");
  vector<vector<float>> vals;
  vector<int> trues;
  // reads file and loads the data
  while (getline(CSV, text)) {
  	dataam++;
  	string strfirstval = text.substr(0, text.find(","));
  	float x = stof(strfirstval);
  	text.erase(0, text.find(",") + 1);
  	string strsecondval = text.substr(0, text.find(","));
  	float y = stof(strsecondval);
  	text.erase(0, text.find(",") + 1);
  	string strthirdval = text;
  	float xy = stof(strthirdval);
  	text.erase(0, text.find(",") + 1);
  	string strfourthval = text;
  	float x2 = stof(strfourthval);
  	text.erase(0, text.find(",") + 1);
  	string strfithval = text;
  	float y2 = stof(strfithval);
  	text.erase(0, text.find(",") + 1);
  	string strsixthval = text;
  	float sinx = stof(strsixthval);
  	text.erase(0, text.find(",") + 1);
  	string strseventhval = text;
  	float siny = stof(strseventhval);
  	text.erase(0, text.find(",") + 1);
  	string streighthval = text;
  	int _true = stoi(streighthval);
  	trues.push_back(_true);
  	vals.push_back({ x, y, xy, x2, y2, sinx, siny });
  }
  // normalises data
  vals = Normalise(vals);
  vals.erase(vals.begin());
  vals.erase(vals.begin());
  // changes vectors into arrays so the neural network can use it
  float** inputs = new float* [dataam];
  float** truevals = new float* [dataam];
  for (int i = 0; i < dataam; i++) {
  	inputs[i] = new float[7];
  	for (int t = 0; t < 7; t++) {
  		inputs[i][t] = vals[i][t];
  	}
  	truevals[i] = new float[2]();
  	if (trues[i] == 0) {
  		truevals[i][0] = 1;
  		truevals[i][1] = 0;
  	}
  	else {
  		truevals[i][0] = 0;
  		truevals[i][1] = 1;
  	}
  }
  // creates network
  Layer testlayer1 = Layer(7, 128, LEAKYRELU, 9e-6, 1e-8, 2e-7, 0.9, 0., 1e-7);
  Layer testlayer2 = Layer(128, 128, LEAKYRELU, 9e-6, 1e-8, 2e-7, 0.9, 0., 1e-7, 5e-3);
  Layer testlayer3 = Layer(128, 2, SOFTMAX, 9e-6, 1e-8, 2e-7, 0.9, 0., 1e-7);
  Network testnetwork = Network(new Layer[3]{ testlayer1, testlayer2, testlayer3 }, 3, BCE, ACCURACY);
  float* accuracies = new float[300];
  // trains network and records accuracies
  for (int i = 0; i < 300; i++) {
  	testnetwork.Train(inputs, truevals, 7, 2, dataam, 1000, 1);
  	float acc = testnetwork.Accuracy(inputs, truevals, 7, 2, dataam);
  	accuracies[i] = acc;
  }
  // stores accuracies
  WriteAccuracy(accuracies, 300);
  return 0;
}
```
