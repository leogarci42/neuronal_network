#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

enum class Activation
{
	SIGMOID,
	RELU,
	SOFTMAX
};
enum class Loss
{
	MSE,
	CROSSENTROPY
};

class Tensor
{
  private:
	std::vector<float> _data;
	size_t _rows;
	size_t _cols;

  public:
	Tensor();
	Tensor(size_t rows, size_t cols);

	float &at(size_t r, size_t c);
	float at(size_t r, size_t c) const;
	float *data();

	size_t rows() const;
	size_t cols() const;
	size_t size() const;

	void randomize(float min, float max);
	void zero();
};

class Layer
{
  private:
	Tensor _w;
	Tensor _b;
	Tensor _out;
	Tensor _in;
	Tensor _gw;
	Tensor _gb;
	Tensor _gin;
	size_t _in_sz;
	size_t _out_sz;
	Activation _act;

	float sigmoid(float x) const;
	float sigmoid_deriv(float x) const;
	float relu(float x) const;
	float relu_deriv(float x) const;
	Tensor &softmax(Tensor &t) const;

  public:
	Layer(size_t in_sz, size_t out_sz, Activation act = Activation::SIGMOID);

	Tensor &forward(const Tensor &in);
	Tensor &backward(const Tensor &grad);
	void update(float lr);
	const Tensor &out() const;
};

class nn
{
  private:
	std::vector<Layer> _layers;
	Tensor _out;
	Loss _loss_fn;

  public:
	nn(Loss loss_fn = Loss::MSE);
	void add(size_t in_sz, size_t out_sz, Activation act = Activation::SIGMOID);
	Tensor &forward(const Tensor &in);
	void backward(const Tensor &exp);
	void train(const Tensor &in, const Tensor &exp, float lr);
	float loss(const Tensor &exp) const;
};