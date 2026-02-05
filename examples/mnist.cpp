#include "../include/nn.hpp"

int	main(void)
{
	nn net(Loss::CROSSENTROPY);

	srand(42);
	net.add(784, 128, Activation::RELU);
	net.add(128, 64, Activation::RELU);
	net.add(64, 10, Activation::SOFTMAX);

	std::cout << "Network ready for MNIST training\n";
	std::cout << "784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)\n";

	return (0);
}