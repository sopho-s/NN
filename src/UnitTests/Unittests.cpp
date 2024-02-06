#include "pch.h"
#include "CppUnitTest.h"
#include "NN.h"
#include <cassert>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace Tests
{
	TEST_CLASS(DotTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float* inputs = new float[2]{ 5, 7 };
			float** weights = new float* [2];
			weights[0] = new float[2]{ 4, 2 };
			weights[1] = new float[2]{ 3, 5 };
			float* eresult = new float[2]{ 41, 45 };
			float* tresult = new float[2];
			tresult = Dot(inputs, weights, 2, 2);
			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
		}
		TEST_METHOD(TestMethod2)
		{
			float* inputs = new float[2]{ 3.4, 2.1 };
			float** weights = new float* [2];
			weights[0] = new float[2]{ 0.2, 0.42 };
			weights[1] = new float[2]{ 0.15, 0.195 };
			float* eresult = new float[2]{ 0.995, 1.8375 };
			float* tresult = new float[2];
			tresult = Dot(inputs, weights, 2, 2);
			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
		}
		TEST_METHOD(TestMethod3)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights = new float* [3];
			weights[0] = new float[3]{ 4, 5, 9 };
			weights[1] = new float[3]{ 2, 3, 2 };
			weights[2] = new float[3]{ 1, 4, 8 };
			float* eresult = new float[3]{ 21, 41, 71 };
			float* tresult = new float[3];
			tresult = Dot(inputs, weights, 3, 3);
			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
	};
	TEST_CLASS(InitialisationLayerTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float** weights = new float* [3];
			float* biases = new float[3]{ 2, 4, -1 };
			weights[0] = new float[3]{ 4, 5, 9 };
			weights[1] = new float[3]{ 2, 3, 2 };
			weights[2] = new float[3]{ 1, 4, 8 };
			float* tresult = new float[3];
			Layer testlayer = Layer(3, 3, weights, biases);
			tresult = testlayer.GetBiases();
			Assert::AreEqual(biases[0], tresult[0]);
			Assert::AreEqual(biases[1], tresult[1]);
			Assert::AreEqual(biases[2], tresult[2]);
		}
		TEST_METHOD(TestMethod2)
		{
			float** weights = new float* [3];
			float* biases = new float[3]{ 2, 4, -1 };
			weights[0] = new float[3]{ 4, 5, 9 };
			weights[1] = new float[3]{ 2, 3, 2 };
			weights[2] = new float[3]{ 1, 4, 8 };
			float** tresult = new float* [3];
			tresult[0] = new float[3];
			tresult[1] = new float[3];
			tresult[2] = new float[3];
			Layer testlayer = Layer(3, 3, weights, biases);
			tresult = testlayer.GetWeights();
			for (int i = 0; i < 3; i++) {
				for (int t = 0; t < 3; t++) {
					Assert::AreEqual(weights[i][t], tresult[i][t]);
				}
			}
		}
	};
	TEST_CLASS(LayerPassTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights = new float* [3];
			float* biases = new float[3]{ 2, 4, -1 };
			weights[0] = new float[3]{ 4, 5, 9 };
			weights[1] = new float[3]{ 2, 3, 2 };
			weights[2] = new float[3]{ 1, 4, 8 };
			float* eresult = new float[3]{ 23, 45, 70 };
			float* tresult = new float[3];
			Layer testlayer = Layer(3, 3, weights, biases);
			tresult = testlayer.Pass(inputs);
			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
		TEST_METHOD(TestMethod2)
		{
			float* inputs = new float[3]{ 3, 2 };
			float** weights = new float* [2];
			float* biases = new float[3]{ 2, 4, -1 };
			weights[0] = new float[3]{ 4, 5, 9 };
			weights[1] = new float[3]{ 2, 3, 2 };
			float* eresult = new float[3]{ 18, 25, 30 };
			float* tresult = new float[3];
			Layer testlayer = Layer(2, 3, weights, biases);
			tresult = testlayer.Pass(inputs);
			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
	};
	TEST_CLASS(NetworkInitialisationTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float** tresult = new float* [3];
			tresult[0] = new float[3];
			tresult[1] = new float[3];
			tresult[2] = new float[3];
			tresult = testnetwork.GetLayerWeights(0);
			for (int i = 0; i < 3; i++) {
				for (int t = 0; t < 3; t++) {
					Assert::AreEqual(weights1[i][t], tresult[i][t]);
				}
			}
			tresult = testnetwork.GetLayerWeights(1);
			for (int i = 0; i < 3; i++) {
				for (int t = 0; t < 3; t++) {
					Assert::AreEqual(weights2[i][t], tresult[i][t]);
				}
			}
		}
		TEST_METHOD(TestMethod2)
		{
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float* tresult = new float[3];
			tresult = testnetwork.GetLayerBiases(0);
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(biases1[i], tresult[i]);
			}
			tresult = testnetwork.GetLayerBiases(1);
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(biases2[i], tresult[i]);
			}
		}

		TEST_METHOD(TestMethod3)
		{
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			int tresult;
			tresult = testnetwork.GetLayerAm();
			Assert::AreEqual(2, tresult);
		}
	};
	TEST_CLASS(NetworkPassTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float* eresult = new float[3]{ 766, 549, 420 };
			float* tresult = new float[3];
			tresult = testnetwork.Pass(inputs);

			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
	};
	TEST_CLASS(NetworkCostTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float* trueval = new float[3]{ 500, 250, 740 };
			float* eresult = new float[3]{ 70756, 89401, 102400 };
			float* tresult = new float[3];
			tresult = testnetwork.Pass(inputs);
			tresult = testnetwork.CalculateCost(trueval);

			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
		TEST_METHOD(TestMethod2)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float* trueval = new float[3]{ 500, 250, 740 };
			float* eresult = new float[3]{ 532, 598, -640 };
			float* tresult = new float[3];
			tresult = testnetwork.Pass(inputs);
			tresult = testnetwork.CalculateDCost(trueval);

			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
		TEST_METHOD(TestMethod3)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float* trueval = new float[3]{ 500, 250, 740 };
			float* eresult = new float[3]{ -64814, 198540, 335580 };
			float* tresult = new float[3];
			testnetwork.Pass(inputs);
			testnetwork.CalculateDCost(trueval);
			testnetwork.BackPropagate();
			tresult = testnetwork.GetLayerCosts(1);

			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
		TEST_METHOD(TestMethod4)
		{
			float* inputs = new float[3]{ 3, 2, 5 };
			float** weights1 = new float* [3];
			float* biases1 = new float[3]{ 2, 4, -1 };
			weights1[0] = new float[3]{ 4, 5, 9 };
			weights1[1] = new float[3]{ 2, 3, 2 };
			weights1[2] = new float[3]{ 1, 4, 8 };
			Layer testlayer1 = Layer(3, 3, weights1, biases1);

			float** weights2 = new float* [3];
			float* biases2 = new float[3]{ 5, -4, 4 };
			weights2[0] = new float[3]{ 2, 1, 7 };
			weights2[1] = new float[3]{ 5, 4, 1 };
			weights2[2] = new float[3]{ 7, 5, 3 };
			Layer testlayer2 = Layer(3, 3, weights2, biases2);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);

			float* trueval = new float[3]{ 500, 250, 740 };
			float* eresult = new float[3]{ 11260992, 2274304, 17069930 };
			float* tresult = new float[3];
			testnetwork.Pass(inputs);
			testnetwork.CalculateDCost(trueval);
			testnetwork.BackPropagate();
			tresult = testnetwork.GetLayerCosts(0);

			Assert::AreEqual(eresult[0], tresult[0]);
			Assert::AreEqual(eresult[1], tresult[1]);
			Assert::AreEqual(eresult[2], tresult[2]);
		}
	};
	TEST_CLASS(NetworkTrainTest)
	{
	public:
		TEST_METHOD(TestMethod1)
		{
			float** inputs = new float* [5]{ new float[1] {0}, new float[1] {1}, new float[1] {-1}, new float[1] {-2}, new float[1] {2} };
			float** truevals = new float* [5]{ new float[1] {3}, new float[1] {5}, new float[1] {1}, new float[1] {-1}, new float[1] {7} };
			Layer testlayer1 = Layer(1, 3);
			Layer testlayer2 = Layer(3, 1);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);
			float firstacc = testnetwork.Accuracy(inputs, truevals, 1, 1, 5);
			testnetwork.Train(inputs, truevals, 1, 1, 5, 1000);
			assert(testnetwork.Accuracy(inputs, truevals, 1, 1, 5) < firstacc);
		}
		TEST_METHOD(TestMethod2)
		{
			float** inputs = new float* [5]{ new float[1] {0}, new float[1] {1}, new float[1] {-1}, new float[1] {-2}, new float[1] {2} };
			float** truevals = new float* [5]{ new float[1] {3}, new float[1] {5}, new float[1] {1}, new float[1] {-1}, new float[1] {7} };
			Layer testlayer1 = Layer(1, 3, RELU, 0.001);
			Layer testlayer2 = Layer(3, 1, RELU, 0.001);
			Network testnetwork = Network(new Layer[2]{ testlayer1, testlayer2 }, 2, MSE);
			float firstacc = testnetwork.Accuracy(inputs, truevals, 1, 1, 5);
			testnetwork.Train(inputs, truevals, 1, 1, 5, 1000);
			float finalacc = testnetwork.Accuracy(inputs, truevals, 1, 1, 5);
			assert(finalacc < firstacc);
		}
	};
}