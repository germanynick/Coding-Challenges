import numpy as np

data = open('kafka.txt', 'r').read()
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)

print('data has %d chars, %d unique' % (data_size, vocab_size))

# Convert from character to integer number
char_to_int = {ch: i for i, ch in enumerate(chars)}

# Convert from integer number to character
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Show Mapping
# print(char_to_int)
# print(int_to_char)


def char_to_vector(char):
  vector_for_char = np.zeros((vocab_size, 1))
  vector_for_char[char_to_int[char]] = 1
  return vector_for_char


# Hyper parameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# Weight
W_i_h = np.random.randn(hidden_size, vocab_size) * 0.01
W_h_h = np.random.randn(hidden_size, hidden_size) * 0.01
W_h_o = np.random.randn(vocab_size, hidden_size) * 0.01

# Bias
b_h = np.zeros((hidden_size, 1))
b_o = np.zeros((vocab_size, 1))


def lossFun(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}  # Empty dicts

  hs[-1] = np.copy(hprev)
  loss = 0
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1))
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(W_i_h, xs[t]) + np.dot(W_h_h, hs[t - 1]) + b_h)

    ys[t] = np.dot(W_h_o, hs[t]) + b_o

    ps[t] = np.exp(ys[t] / np.sum(np.exp(ys[t])))

    loss += -np.log(ps[t][targets[t], 0])  # Softmax (cross-entropy)

  # Weight
  dW_i_h, dW_h_h, dW_h_o = np.zeros_like(
      W_i_h), np.zeros_like(W_h_h), np.zeros_like(W_h_o)

  # Bias
  db_h, db_o = np.zeros_like(b_h), np.zeros_like(b_o)
  dh_next = np.zeros_like(hs[0])

  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1

    dW_h_o += np.dot(dy, hs[t].T)

    db_o += dy
    dh = np.dot(W_h_o.T, dy) + dh_next

    dhraw = (1 - hs[t] * hs[t]) * dh

    db_h += dhraw
    dW_i_h += np.dot(dhraw, xs[t].T)
    dW_h_h += np.dot(dhraw, hs[t - 1].T)
    dh_next = np.dot(W_h_h.T, dhraw)

  for dparam in [dW_i_h, dW_h_h, dW_h_o, db_h, db_o]:
    np.clip(dparam, -5, 5, out=dparam)

  return loss, dW_i_h, dW_h_h, dW_h_o, db_h, db_o, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1

  ixes = []

  for t in range(n):
    h = np.tanh(np.dot(W_i_h, x) + np.dot(W_h_h, h) + b_h)
    y = np.dot(W_h_o, h) + b_o

    p = np.exp(y) / np.sum(np.exp(y))

    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1

    ixes.append(ix)

  txt = ''.join(int_to_char[ix] for ix in ixes)
  print('----\n %s \n----' % (txt))


hprev = np.zeros((hidden_size, 1))
sample(hprev, char_to_int['a'], 200)

p = 0
inputs = [char_to_int[ch] for ch in data[p:p + seq_length]]
print("inputs", inputs)
targets = [char_to_int[ch] for ch in data[p + 1:p + seq_length + 1]]
print("targets", targets)

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(W_i_h), np.zeros_like(W_h_h), np.zeros_like(W_h_o)
mbh, mby = np.zeros_like(b_h), np.zeros_like(b_o) # memory variables for Adagrad                                                                                                                
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0                                                                                                                        
while n<=1000*100:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  # check "How to feed the loss function to see how this part works
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory                                                                                                                                      
    p = 0 # go from start of data                                                                                                                                                             
  inputs = [char_to_int[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_int[ch] for ch in data[p+1:p+seq_length+1]]

  # forward seq_length characters through the net and fetch gradient                                                                                                                          
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  # sample from the model now and then                                                                                                                                                        
  if n % 1000 == 0:
    print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
    sample(hprev, inputs[0], 200)

  # perform parameter update with Adagrad                                                                                                                                                     
  for param, dparam, mem in zip([W_i_h, W_h_h, W_h_o, b_h, b_o],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update                                                                                                                   

  p += seq_length # move data pointer                                                                                                                                                         
  n += 1 # iteration counter