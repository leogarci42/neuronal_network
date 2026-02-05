CC = c++
NAME = libnn.a
MNIST_NAME = mnist
XOR_NAME = xor
CFLAGS = -Wall -Wextra -Werror -std=c++11 -I./include
OBJ_DIR = obj
SRC_DIR = src
EXAMPLE_DIR = examples

SRC = nn.cpp
MNIST_SRC = mnist.cpp
XOR_SRC = xor.cpp

OBJ = $(addprefix $(OBJ_DIR)/$(SRC_DIR)/, $(SRC:.cpp=.o))
MNIST_OBJ = $(addprefix $(OBJ_DIR)/$(EXAMPLE_DIR)/, $(MNIST_SRC:.cpp=.o))
XOR_OBJ = $(addprefix $(OBJ_DIR)/$(EXAMPLE_DIR)/, $(XOR_SRC:.cpp=.o))
DIRS = $(sort $(dir $(OBJ) $(MNIST_OBJ) $(XOR_OBJ)))

all: $(NAME)

$(NAME): $(OBJ)
	@echo "$(BLUE)Creating library $(NAME)...$(RESET)"
	@ar rcs $(NAME) $(OBJ)
	@echo "$(GREEN)Library created successfully!$(RESET)"

mnist: $(OBJ) $(MNIST_OBJ)
	@echo "$(BLUE)Compiling $(MNIST_NAME)...$(RESET)"
	@$(CC) $(CFLAGS) $(OBJ) $(MNIST_OBJ) -o $(MNIST_NAME)
	@echo "$(GREEN)$(MNIST_NAME) created successfully!$(RESET)"

xor: $(OBJ) $(XOR_OBJ)
	@echo "$(BLUE)Compiling $(XOR_NAME)...$(RESET)"
	@$(CC) $(CFLAGS) $(OBJ) $(XOR_OBJ) -o $(XOR_NAME)
	@echo "$(GREEN)$(XOR_NAME) created successfully!$(RESET)"

$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp | $(DIRS)
	@echo "$(YELLOW)Compiling $<...$(RESET)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/$(EXAMPLE_DIR)/%.o: $(EXAMPLE_DIR)/%.cpp | $(DIRS)
	@echo "$(YELLOW)Compiling $<...$(RESET)"
	@$(CC) $(CFLAGS) -c $< -o $@

$(DIRS):
	@mkdir -p $@

clean:
	@echo "$(RED)Cleaning object files...$(RESET)"
	@rm -rf $(OBJ_DIR)

fclean: clean
	@echo "$(RED)Cleaning executables and library...$(RESET)"
	@rm -f $(NAME) $(MNIST_NAME) $(XOR_NAME)

re: fclean all

.PHONY: all clean fclean re mnist xor $(DIRS)

RESET  = \033[0m
RED    = \033[31m
GREEN  = \033[32m
YELLOW = \033[33m
BLUE   = \033[34m
BOLD   = \033[1m