# 5. Neuronové sítě

> Vícevrstvé sítě a jejich výrazové schopnosti. Učení neuronových sítí: Gradientní sestup, zpětná propagace, praktické otázky učení (příprava dat, inicializace vah, volba a adaptace hyperparametrů). Regularizace. Konvoluční sítě. Rekurentní sítě (LSTM). (PV021)


# Neural Networks

## 1. Multilayer Networks and Their Expressive Power
Multilayer networks, particularly Multilayer Perceptrons (MLPs), are fundamental architectures in neural networks. They consist of an input layer, one or more hidden layers, and an output layer, with neurons connected sequentially.

### Architecture
- **Feedforward Network**: Connections flow in one direction; there are no cycles.
- **Multilayer Perceptron (MLP)**: Neurons are organized in layers, fully connected between consecutive layers. The architecture can be represented as, e.g., `2-4-3-2` for a network with:
  - 2 input neurons,
  - 4 neurons in the first hidden layer,
  - 3 neurons in the second hidden layer,
  - 2 output neurons.

### Expressive Power
The expressive power of neural networks lies in their ability to approximate any continuous function. For example:
- A two-layer MLP with the unit step activation function \(\sigma(\xi) = \begin{cases} 
1 & \text{if } \xi \geq 0 \\
0 & \text{if } \xi < 0 
\end{cases}\)
can compute any Boolean function \(F : \{0,1\}^n \to \{0,1\}\).

### Non-Linear Separation
- Hidden layers divide the input space into half-spaces.
- Subsequent layers combine these half-spaces into convex sets and unions of convex sets, enabling complex decision boundaries.

## 2. Learning in Neural Networks
Learning in neural networks involves optimizing the weights to minimize the error function. This is achieved using:

### Gradient Descent
Weights \(w_i\) are updated iteratively using:
\[
w_i \leftarrow w_i - \eta \frac{\partial E}{\partial w_i}
\]
where:
- \(\eta\) is the learning rate,
- \(E\) is the error function (e.g., mean squared error, cross-entropy).

### Backpropagation
Backpropagation computes the gradient of \(E\) efficiently using:
1. **Forward Pass**: Compute activations layer by layer.
2. **Backward Pass**: Calculate errors and gradients layer by layer.

### Practical Considerations
1. **Data Preparation**: Normalize input features to improve convergence.
2. **Weight Initialization**: Use methods like Xavier or He initialization.
3. **Hyperparameter Tuning**: Adjust learning rates, batch sizes, and epochs.
4. **Regularization**: Apply techniques like dropout or L2 regularization to avoid overfitting.

## 3. Regularization
Regularization enhances the generalization capability of neural networks by:
- **Dropout**: Randomly disabling a fraction of neurons during training.
- **L2 Regularization**: Penalizing large weights by adding a term to the loss function:
\[
E' = E + \lambda \sum_{i} w_i^2
\]

## 4. Convolutional Neural Networks (CNNs)
CNNs are specialized for grid-like data such as images. Key components include:
- **Convolutional Layers**: Extract spatial features using filters \(W\):
\[
z = W * x + b
\]
- **Pooling Layers**: Reduce dimensions (e.g., max pooling).
- **Fully Connected Layers**: Combine extracted features for final output.

Applications: Image classification, object detection, and semantic segmentation.

## 5. Recurrent Neural Networks (RNNs) and LSTMs
### RNNs
RNNs are designed for sequential data, maintaining a hidden state \(h_t\):
\[
h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b)
\]
However, they struggle with long-term dependencies due to vanishing gradients.

### LSTMs
LSTMs address this with:
1. **Forget Gate**: Controls what information to discard:
\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]
2. **Input Gate**: Updates cell state with new input:
\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]
\[
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\]
\[
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
\]
3. **Output Gate**: Generates the next hidden state:
\[
h_t = o_t \cdot \tanh(C_t)
\]

Applications: Time-series prediction, language modeling, speech recognition.

## 6. Activation Functions
Activation functions introduce non-linearities, enabling networks to model complex patterns. Examples:
- **Sigmoid**:
\[
\sigma(\xi) = \frac{1}{1 + e^{-\lambda \xi}}
\]
- **ReLU**:
\[
\sigma(\xi) = \max(0, \xi)
\]
- **Hyperbolic Tangent**:
\[
\sigma(\xi) = \frac{e^{\xi} - e^{-\xi}}{e^{\xi} + e^{-\xi}}
\]

## 7. Key Features of Neural Networks
- **Massive Parallelism**: Neurons process inputs simultaneously.
- **Learning Capability**: Adjust weights to model functions.
- **Generalization**: Perform well on unseen data.
- **Robustness**: Handle noisy inputs and graceful degradation.
