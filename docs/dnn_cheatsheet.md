# Deep Learning & DNN Cheat Sheet

## 1. Activation Functions

- **What is an activation?**
  - The output of a neuron after applying a non-linear function.
- **Why is it called “activation”?**
  - Inspired by biological neurons — determines if a neuron "fires."

### Common Activation Functions

| Name  | Formula             | Notes                        |
|-------|---------------------|------------------------------|
| ReLU  | `max(0, x)`         | Fast, introduces sparsity    |
| Sigmoid | `1 / (1 + e^-x)`  | Good for binary probs, saturates |
| Tanh  | `(e^x - e^-x)/(e^x + e^-x)` | Zero-centered version of sigmoid |
| GELU  | smoother variant of ReLU | Used in Transformers        |

---

## 2. Training Dynamics

### Batch Size vs Epochs

- **Batch size**: How many samples per training step
- **Epochs**: Full passes through the training set

### Learning Rate (`lr`)

- Controls the size of updates during training
- Typical values: 0.1, 0.01, 0.001

### Weight Decay

- Also called L2 regularization
- Penalizes large weights to reduce overfitting

### Dropout

- Randomly disables a fraction of neurons during training
- Typical values: 0.1 to 0.5
- Applied **per-layer**, not globally

---

## 3. Normalization Layers

### BatchNorm

- Normalizes across the **batch**
- More stable training, used in CNNs

### LayerNorm

- Normalizes across the **features** of a single sample
- Better for RNNs, Transformers

---

## 4. Bias & Variance

- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)

### Bias-Variance Tradeoff

| Model Type     | Bias | Variance |
|----------------|------|----------|
| Underfit       | High | Low      |
| Overfit        | Low  | High     |
| Just right     | Low  | Low      |

---

## 5. Transformers Concepts

### Query, Key, Value

- **Query (Q)**: What I’m looking for
- **Key (K)**: What each word offers
- **Value (V)**: The content if that word is attended to

### Attention Score

- Attention = softmax(Q · Kᵀ / √d_k) · V
- Uses **dot product** for similarity
- Softmax normalizes into weights

---

## 6. Softmax vs Normalization

- Softmax is **not** the same as z-score normalization
- Converts logits into a probability distribution

---

## 7. Model Evaluation

### Accuracy vs F1

- **Accuracy**: Overall correct predictions
- **Macro F1**: Average F1 across all classes (equal weight)

### When they’re close?
- Data is balanced
- Model performs uniformly well

---

## 8. Model Saving / Eval Practices

- Save model + vocab together
- Eval should use **same vocab** as training
- Track `<unk>` token usage on test set

---

## 9. Performance Benchmarks (AG News)

| Model               | Accuracy |
|---------------------|----------|
| Your DNN            | 87.5%    |
| FastText            | 91.5%    |
| CNN (Kim-style)     | 91–92%   |
| BERT (finetuned)    | 94–95%   |

---

## 10. Optimization Workflow

- Train/val split
- Grid search on:
  - Learning rate
  - Dropout
  - Hidden size
- Use validation score to pick best model
- Final test evaluation only once
