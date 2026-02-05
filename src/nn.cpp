#include "../include/nn.hpp"

nn::nn(Loss loss_fn) : _loss_fn(loss_fn)
{
}

Tensor::Tensor() : _rows(0), _cols(0)
{
}

Tensor::Tensor(size_t rows, size_t cols) : _data(rows * cols, 0.0f),
	_rows(rows), _cols(cols)
{
}

float &Tensor::at(size_t r, size_t c)
{
	return (_data[r * _cols + c]);
}

float Tensor::at(size_t r, size_t c) const
{
	return (_data[r * _cols + c]);
}

float *Tensor::data()
{
	return (_data.data());
}

size_t Tensor::rows() const
{
	return (_rows);
}

size_t Tensor::cols() const
{
	return (_cols);
}

size_t Tensor::size() const
{
	return (_data.size());
}

void Tensor::randomize(float min, float max)
{
	float	r;

	for (size_t i = 0; i < _data.size(); i++)
	{
		r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		_data[i] = min + r * (max - min);
	}
}

void Tensor::zero()
{
	for (size_t i = 0; i < _data.size(); i++)
		_data[i] = 0.0f;
}

float Layer::sigmoid(float x) const
{
	return (1.0f / (1.0f + std::exp(-x)));
}

float Layer::sigmoid_deriv(float x) const
{
	float s = sigmoid(x);
	return (s * (1.0f - s));
}

float Layer::relu(float x) const
{
	return (x > 0.0f ? x : 0.0f);
}

float Layer::relu_deriv(float x) const
{
	return (x > 0.0f ? 1.0f : 0.0f);
}

Tensor &Layer::softmax(Tensor &t) const
{
	float max_val = t.at(0, 0);
	for (size_t j = 1; j < t.cols(); j++)
		if (t.at(0, j) > max_val)
			max_val = t.at(0, j);

	float sum = 0.0f;
	for (size_t j = 0; j < t.cols(); j++)
	{
		t.at(0, j) = std::exp(t.at(0, j) - max_val);
		sum += t.at(0, j);
	}
	for (size_t j = 0; j < t.cols(); j++)
		t.at(0, j) /= sum;
	return (t);
}

Layer::Layer(size_t in_sz, size_t out_sz, Activation act) : _w(in_sz, out_sz),
	_b(1, out_sz), _out(1, out_sz), _in(1, in_sz), _gw(in_sz, out_sz), _gb(1,
	out_sz), _gin(1, in_sz), _in_sz(in_sz), _out_sz(out_sz), _act(act)
{
	_w.randomize(-1.0f, 1.0f);
	_b.randomize(-0.1f, 0.1f);
}

Tensor &Layer::forward(const Tensor &in)
{
	float	sum;

	for (size_t i = 0; i < _in_sz; i++)
		_in.at(0, i) = in.at(0, i);
	for (size_t j = 0; j < _out_sz; j++)
	{
		sum = 0.0f;
		for (size_t i = 0; i < _in_sz; i++)
			sum += in.at(0, i) * _w.at(i, j);
		sum += _b.at(0, j);
		if (_act == Activation::SIGMOID)
			_out.at(0, j) = sigmoid(sum);
		else if (_act == Activation::RELU)
			_out.at(0, j) = relu(sum);
		else
			_out.at(0, j) = sum;
	}
	if (_act == Activation::SOFTMAX)
		softmax(_out);
	return (_out);
}

Tensor &Layer::backward(const Tensor &grad)
{
	float	z;
	float	d;

	_gin.zero();
	_gw.zero();
	_gb.zero();
	for (size_t j = 0; j < _out_sz; j++)
	{
		z = 0.0f;
		for (size_t i = 0; i < _in_sz; i++)
			z += _in.at(0, i) * _w.at(i, j);
		z += _b.at(0, j);
		if (_act == Activation::SIGMOID)
			d = grad.at(0, j) * sigmoid_deriv(z);
		else if (_act == Activation::RELU)
			d = grad.at(0, j) * relu_deriv(z);
		else
			d = grad.at(0, j);
		_gb.at(0, j) = d;
		for (size_t i = 0; i < _in_sz; i++)
		{
			_gw.at(i, j) = _in.at(0, i) * d;
			_gin.at(0, i) += _w.at(i, j) * d;
		}
	}
	return (_gin);
}

void Layer::update(float lr)
{
	for (size_t i = 0; i < _in_sz; i++)
		for (size_t j = 0; j < _out_sz; j++)
			_w.at(i, j) -= lr * _gw.at(i, j);
	for (size_t j = 0; j < _out_sz; j++)
		_b.at(0, j) -= lr * _gb.at(0, j);
}

const Tensor &Layer::out() const
{
	return (_out);
}

void nn::add(size_t in_sz, size_t out_sz, Activation act)
{
	_layers.emplace_back(in_sz, out_sz, act);
}

Tensor &nn::forward(const Tensor &in)
{
	const Tensor	*cur = &in;

	for (size_t i = 0; i < _layers.size(); i++)
		cur = &_layers[i].forward(*cur);
	_out = *cur;
	return (_out);
}

void nn::backward(const Tensor &exp)
{
	Tensor	*cg;

	Tensor g(1, exp.cols());
	for (size_t j = 0; j < exp.cols(); j++)
		g.at(0, j) = _out.at(0, j) - exp.at(0, j);
	cg = &g;
	for (int i = _layers.size() - 1; i >= 0; i--)
		cg = &_layers[i].backward(*cg);
}

void nn::train(const Tensor &in, const Tensor &exp, float lr)
{
	forward(in);
	backward(exp);
	for (size_t i = 0; i < _layers.size(); i++)
		_layers[i].update(lr);
}

float nn::loss(const Tensor &exp) const
{
	if (exp.cols() == 0)
		return (std::cerr << "Error: Expected tensor has zero columns\n", 0.0f);

	float sum = 0.0f;
	const float MAX_LOSS = 1e6f;

	if (_loss_fn == Loss::CROSSENTROPY)
	{
		for (size_t j = 0; j < exp.cols(); j++)
		{
			float pred = _out.at(0, j);
			if (pred < 1e-7f)
				pred = 1e-7f;
			if (pred > 1.0f - 1e-7f)
				pred = 1.0f - 1e-7f;
			sum -= exp.at(0, j) * std::log(pred);
		}
	}
	else
	{
		for (size_t j = 0; j < exp.cols(); j++)
		{
			float d = _out.at(0, j) - exp.at(0, j);
			sum += d * d;
		}
		sum /= exp.cols();
	}
	return (sum > MAX_LOSS ? MAX_LOSS : sum);
}