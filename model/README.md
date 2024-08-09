# Neural Network

Input layer: $X = H^0$

Hidden layer: $ H^1, H^2, H^3 \dotsm H^k $ (For each bias: $B^1, B^2, B^3 \dotsm B^k$)

Weight matrix: $W^1, W^2, W^3 \dotsm W^k, W^{k+1}$

Output layer: $Y=H^{k+1}$

Target: $T$

Error: $E = error(T, Y)$

With FeedForward, we can express $H^{i+1} = activate(W^i \cdot H^i + B^i) = activate(U^{i+1})$

In order to reduce error, with gradient descent method, we need to update weight like below.

$$
W_ {new} = W_ {old} - \eta \cdot {\partial E \over \partial W}
$$

Let us scope to layer of hidden to output.

$Y = H^{k+1}= activate(U^{k+1}) = activate(H^k W^k + B^k)$

When we apply chain rule, we can get $W_{new}^{k} = W_{old}^{k} - \eta \cdot {\partial E \over \partial H^{k+1}} \cdot {\partial H^{k+1} \over \partial U^{k+1}} \cdot {\partial U^{k+1} \over \partial W^{k}} $

$$
{\partial E \over \partial H^{k+1}} = {\partial E \over \partial Y} = {\partial error(T, Y) \over \partial Y}
$$

$$
{\partial E \over \partial U^{k+1}} = {\partial E \over \partial H^{k+1}} \cdot {\partial H^{k+1} \over \partial U^{k+1}} = {\partial E \over \partial H^{k+1}} \cdot {\partial activate(U^{k+1}) \over \partial U^{k+1}}
$$

$$
{\partial E \over \partial W^{k+1}} = {\partial E \over \partial U^{k+1}} \cdot {\partial U^{k+1} \over \partial W^{k}} = {\partial E \over \partial U^{k+1}} \cdot {\partial (H^{k} W^k + B^{k}) \over \partial W^k} = {\partial E \over \partial U^{k+1}} \cdot H^k
$$

And let's move to immediately previous layer.

$$
{\partial E \over \partial H^{k}} = {\partial E \over \partial U^{k+1}} \cdot {\partial U^{k+1} \over \partial H^{k}} = {\partial E \over \partial U^{k+1}} \cdot {\partial (H^k W^k + B^k) \over \partial H^k}
$$


$$
{\partial E \over \partial H^{k}} = {\partial E \over \partial U^{k+1}} \cdot {\partial U^{k+1} \over \partial H^{k}}
$$
