import copy, numpy as np

np.random.seed(0)

def sigmoid(x, dev = False):
  return x * (1 - x) if dev else 1/(1 + np.exp(-x))

int2binary = {}
binary_dim = 8

largest_num = pow(2, binary_dim)

binary =  np.unpackbits(
  np.array([range(largest_num)], dtype=np.uint8).T, 
  axis=1
)

for i in range(largest_num):
  int2binary[i] = binary[i]

alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# Weight && bias
W_i = 2 * np.random.random((input_dim, hidden_dim)) - 1
W_o = 2 * np.random.random((hidden_dim, output_dim)) - 1
W_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

W_i_u = np.zeros_like(W_i)
W_o_u = np.zeros_like(W_o)
W_h_u = np.zeros_like(W_h)

# Training logic
for j in range(10000):
  # Generate a simple addition problem (a + b = c)
  a_int = np.random.randint(largest_num / 2)
  a_bin = int2binary[a_int]

  b_int = np.random.randint(largest_num / 2)
  b_bin = int2binary[b_int]

  # c = a + b
  c_int = a_int + b_int
  c_bin = int2binary[c_int]

  # Predict output
  guess = np.zeros_like(c_bin)

  error = 0

  for position in range(binary_dim)
    # Input
    input = np.array([])
