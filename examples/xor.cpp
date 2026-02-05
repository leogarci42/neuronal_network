#include "../include/nn.hpp"

int	main(void)
{
	nn		net;
	float	inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	float	targets[4][1] = {{0}, {1}, {1}, {0}};
	float	total_loss;

	srand(42);
	net.add(2, 4);
	net.add(4, 1);
	std::cout << "Training XOR...\n\n";
	for (int epoch = 0; epoch <= 10000; epoch++)
	{
		total_loss = 0.0f;
		for (int i = 0; i < 4; i++)
		{
			Tensor in(1, 2);
			in.at(0, 0) = inputs[i][0];
			in.at(0, 1) = inputs[i][1];
			Tensor exp(1, 1);
			exp.at(0, 0) = targets[i][0];
			net.train(in, exp, 0.5f);
			total_loss += net.loss(exp);
		}
		if (epoch % 1000 == 0)
			std::cout << "Epoch " << epoch << " | Loss: " << total_loss
				/ 4 << "\n";
	}
	std::cout << "\nResults:\n";
	for (int i = 0; i < 4; i++)
	{
		Tensor in(1, 2);
		in.at(0, 0) = inputs[i][0];
		in.at(0, 1) = inputs[i][1];
		Tensor &out = net.forward(in);
		std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << out.at(0,
			0) << " (expected " << targets[i][0] << ")\n";
	}
	return (0);
}
