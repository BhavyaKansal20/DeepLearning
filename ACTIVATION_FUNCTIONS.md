# ⚡ Types of Activation Functions

> **Deep Learning Series** &nbsp;|&nbsp; 🧠 Neural Network Fundamentals &nbsp;|&nbsp; 📚 Complete Reference Guide

---

## 🔍 What is an Activation Function?

An **Activation Function** decides whether a neuron should be activated or not — it introduces **non-linearity** into the network, enabling it to learn complex patterns that a purely linear model never could.

```
INPUT ──► WEIGHTED SUM ──► ACTIVATION FUNCTION ──► OUTPUT
               (Σ wᵢxᵢ + b)         f(a)
```

---

## 📋 Quick Comparison Table

| # | Function | Formula | Range | Best Used In |
|:---:|---|---|:---:|---|
| 1 | **Linear** | $f(x) = x$ | $(-\infty, +\infty)$ | Output (Regression) |
| 2 | **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Output (Binary Classif.) |
| 3 | **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, +1)$ | Hidden Layers |
| 4 | **ReLU** | $\max(0, x)$ | $[0, +\infty)$ | Hidden Layers (CNN/DNN) |
| 5 | **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, +\infty)$ | Deep Networks |
| 6 | **PReLU** | $\max(\alpha x, x)$ | $(-\infty, +\infty)$ | CNNs, Deep Networks |
| 7 | **Swish** | $x \cdot \sigma(x)$ | $(-\infty, +\infty)$ | Deep Networks (Google) |

---

## 1️⃣ Linear Activation Function

> *The simplest type — output is exactly equal to the input.*

$$\boxed{f(x) = x}$$

The output is the **same as the input**. The neuron doesn't transform the data — it just passes it forward as-is.

![Linear Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20190901213212/Screenshot-2019-09-01-at-9.32.01-PM.png)

*Figure 1 — Linear activation: a straight diagonal line with slope = 1*

---

### 📌 Where Do We Use It?

- **Regression tasks** — predicting continuous values like salary, house price, temperature, etc.
- **Output layer** of a neural network when we don't want output restricted to 0–1 (sigmoid) or −1 to +1 (tanh)

> **Example:** Predicting house prices — you want outputs like ₹50,00,000 or ₹80,00,000.
> A sigmoid would squeeze everything between 0 and 1, which makes no sense here. ✅

---

### ⚠️ Limitations

| Problem | Explanation |
|---|---|
| **Linear Model Collapse** | If you use linear activation in all layers, the whole network becomes just a linear model — no matter how many layers you add |
| **Cannot Capture Complexity** | Cannot learn complex, non-linear patterns in data |

> 💡 **That's why** hidden layers use non-linear activations like **ReLU, tanh, sigmoid** — but the output layer for regression *can* be linear.

---
---

## 2️⃣ Sigmoid Activation Function

> *An S-shaped curve that squashes any real number into a range between 0 and 1.*

$$\boxed{f(x) = \frac{1}{1 + e^{-x}}}$$

No matter how large or small the input — **output always stays between 0 and 1**.

![Sigmoid Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20190901213157/Screenshot-2019-09-01-at-9.31.45-PM.png)

*Figure 2 — Sigmoid (blue) and its derivative (red): classic S-curve between 0 and 1*

---

### 📌 Where Do We Use It?

- **Binary classification** problems (e.g., predicting yes/no, disease/no disease, spam/not spam)
- **Output layer** when you want a **probability** as the output

> **Example:** If the sigmoid outputs `0.85`, you can interpret it as **85% chance of having heart disease**. ✅

---

### ⚠️ Limitations

| Problem | Explanation |
|---|---|
| **Vanishing Gradient** | For very large or very small inputs, gradient becomes almost 0 — slows learning badly |
| **Not Preferred in Hidden Layers** | ReLU is preferred in hidden layers nowadays for this reason |

---
---

## 3️⃣ Tanh Activation Function

> *Like sigmoid, but centered at 0 — squeezes values into −1 to +1.*

The **tanh** (hyperbolic tangent) is another squashing function. Instead of 0 to 1, it squeezes values into **−1 to +1**.

$$\boxed{f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}}$$

![Tanh Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20190901215947/Screenshot-2019-09-01-at-9.59.37-PM.png)

*Figure 3 — Tanh activation: S-curve centered at 0, ranging from −1 to +1*

---

### 📌 Where Do We Use It?

- **Hidden layers** of neural networks
- Useful when data has **both positive and negative values** — it centres the output around 0 (unlike sigmoid which is centred at 0.5)

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Outputs are **zero-centred** (good for optimization) | Still suffers from **vanishing gradient** when inputs are very large/negative |
| **Stronger gradients** than sigmoid in range (−1, 1) → learning can be faster | That's why ReLU is more common in hidden layers in modern deep learning |

---
---

## 4️⃣ ReLU Activation Function

> *The most widely used activation function in modern deep learning.*

**ReLU** stands for **Rectified Linear Unit**. It's super simple:

$$\boxed{f(x) = \max(0, x)}$$

**That means:**

```
If input x < 0  →  output = 0
If input x > 0  →  output = x
```

It **passes positive values as-is** and **blocks negative values** by turning them to 0.

![ReLU Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20190901213256/Screenshot-2019-09-01-at-9.32.21-PM.png)

*Figure 4 — ReLU: flat at 0 for negatives, linear for positives — the "hockey stick" curve*

---

### 📌 Where Do We Use It?

- **Hidden layers** of almost all modern deep neural networks
- Works really well in **CNNs** (Convolutional Neural Networks), image recognition, NLP, and many more tasks

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Very **fast and simple** to compute | **Dying ReLU problem** — neurons can get stuck at 0 forever if weights update badly |
| Helps **avoid vanishing gradient** problem (better than sigmoid/tanh) | **Not smooth at 0** — not differentiable there, but still works fine in practice |
| Makes training **deep networks much faster** | — |

---
---

## 5️⃣ Leaky ReLU Activation Function

> *ReLU with a small fix — neurons never completely die.*

It's just like **ReLU, but with a small twist**. In ReLU, whenever input is negative → output is 0.

**Leaky ReLU's fix:** Instead of giving 0 for negative inputs, it gives a **tiny negative value** (like `0.01 × input`). This way, the neuron is **never completely dead**.

$$\boxed{f(x) = \begin{cases} x & \text{if } x > 0 \\ 0.01x & \text{if } x \leq 0 \end{cases}}$$

![Leaky ReLU vs ReLU](https://media.geeksforgeeks.org/wp-content/uploads/20190901213559/Screenshot-2019-09-01-at-9.35.54-PM.png)

*Figure 5 — Leaky ReLU (orange) vs ReLU (blue): small negative slope instead of flat zero*

---

### 📌 Advantages of Leaky ReLU

**🔧 Fixes "Dead Neuron" Problem**
- In normal ReLU, if inputs go negative, output is always 0 — sometimes the neuron stops learning permanently (dead neuron)
- Leaky ReLU solves this by allowing a **small negative slope**, so neurons still update weights

**⚡ Computationally Simple**
- Just like ReLU, the function is very easy to compute (no heavy math like exponentials in Sigmoid/Tanh)

**📈 Better Gradient Flow**
- Since even negative inputs have a small gradient (e.g., 0.01), the network can **continue learning**, reducing the vanishing gradient issue

**🏗️ Works Well in Deep Networks**
- Especially useful in deep neural networks where ReLU may suffer from many dead neurons

---

### ⚠️ Limitations

- Small negative slope may **bias results**
- Slope value **needs tuning** — the `0.01` is a hyperparameter

---
---

## 6️⃣ PReLU Activation Function

> *Leaky ReLU, but smarter — the slope is learned, not fixed.*

**PReLU** (Parametric Rectified Linear Unit) is an **improved version of Leaky ReLU**:

- In **Leaky ReLU** → slope is fixed by us (e.g., `0.01`)
- In **PReLU** → slope `α` is **learned automatically** by the model during training → more flexible and adaptive

$$\boxed{f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}}$$

*where **α** is a trainable parameter*

![PReLU vs Leaky ReLU vs ReLU](https://media.geeksforgeeks.org/wp-content/uploads/20190901215230/Screenshot-2019-09-01-at-9.52.20-PM.png)

*Figure 6 — Comparison of ReLU, Leaky ReLU, and PReLU — slope α is adaptive in PReLU*

---

### 🧠 Intuition (Easy Way)

```
ReLU:       Negative values are KILLED        (output = 0)
Leaky ReLU: Negative values get a tiny leak   (e.g., 0.01x)
PReLU:      Instead of fixing that leak, the model says
            "I'll learn the best leak slope myself." 🤖
```

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| Fixes dead neurons (like Leaky ReLU) | Extra parameters — slope `α` adds more trainable values |
| **Adaptive** — slope is learned, not fixed | Risk of **overfitting** if dataset is small |
| Better accuracy — often improves CNNs and deep networks | Slightly more complex than plain ReLU |

---
---

## 7️⃣ Swish Activation Function

> *Smooth, non-linear, and introduced by Google — often outperforms ReLU.*

**Swish** is a smooth, non-linear activation function introduced by **Google researchers**.

$$\boxed{f(x) = x \cdot \sigma(x)}$$

Where $\sigma(x)$ is the **sigmoid function**. So basically:

$$\text{Swish} = x \times \text{Sigmoid}(x)$$

![Swish Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20190901215918/Screenshot-2019-09-01-at-9.59.13-PM.png)

*Figure 7 — Swish (blue), its first derivative, and second derivative — smooth curve around zero*

---

### 🧠 Intuition (Easy Way)

> Think of it as **ReLU but smoother.**

```
For large positive inputs  →  output ≈ input (like ReLU) ✅
For large negative inputs  →  output is small but NOT strictly zero (like Leaky ReLU) ✅
Around zero               →  the curve is SMOOTH, not sharp like ReLU ✅
```

This smoothness often makes training deep networks **easier and more stable**.

---

### ✅ Advantages vs ⚠️ Limitations

| ✅ Advantages | ⚠️ Limitations |
|---|---|
| **Smooth curve** → better gradient flow, avoids sharp jumps like ReLU | **More computation** needed (requires sigmoid) |
| **Non-monotonic** → can adapt better to complex patterns | Not always better than ReLU (depends on problem) |
| Works well in deep networks — often **improves accuracy over ReLU** | Slight risk of **slower training** compared to simple ReLU |

---
---

## 🔮 Master Comparison — All 7 Functions

```
OUTPUT
  │
1 │         ····················· Sigmoid
  │        ·
  │       · ╱──────────────────── ReLU / Linear (x > 0)
  │      ·╱
0 │─────╳────────────────────────── x
  │   ╱· ╲
  │  ╱  ·  ╲──────────────────── Leaky ReLU (slight slope for x < 0)
  │ ╱    ···
-1│       ····················· Tanh
  │
  └──────────────────────────────────────► INPUT x
        negative        positive
```

---

## 📐 All Formulas — Quick Reference Card

$$\boxed{\text{Linear:} \quad f(x) = x}$$

$$\boxed{\text{Sigmoid:} \quad f(x) = \frac{1}{1+e^{-x}}}$$

$$\boxed{\text{Tanh:} \quad f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}}$$

$$\boxed{\text{ReLU:} \quad f(x) = \max(0, x)}$$

$$\boxed{\text{Leaky ReLU:} \quad f(x) = \begin{cases} x & x > 0 \\ 0.01x & x \leq 0 \end{cases}}$$

$$\boxed{\text{PReLU:} \quad f(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases} \quad (\alpha \text{ is trainable})}$$

$$\boxed{\text{Swish:} \quad f(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}}$$

---

## 🎯 How to Choose the Right Activation Function

```
Is this the OUTPUT LAYER?
│
├── YES → Regression?      ──► Linear
│         Binary Classif.? ──► Sigmoid
│         Multi-class?     ──► Softmax
│
└── NO (Hidden Layer)
    │
    ├── Start with          ──► ReLU  (fast, simple, works great)
    │
    ├── Dead Neurons?       ──► Leaky ReLU or PReLU
    │
    ├── Need adaptability?  ──► PReLU (α is learned)
    │
    └── Want state-of-art?  ──► Swish (Google's choice for deep nets)
```

---

## ⚠️ The Vanishing Gradient Problem — Visual Summary

| Activation | Gradient Behavior | Verdict |
|---|---|:---:|
| **Sigmoid** | Gradient → ~0 for large/small inputs | ❌ Bad in deep nets |
| **Tanh** | Better than sigmoid, but still vanishes | ⚠️ OK in shallow nets |
| **ReLU** | Gradient = 1 for x > 0, 0 for x < 0 | ✅ Mostly Good |
| **Leaky ReLU** | Small gradient even for x < 0 | ✅ Better |
| **PReLU** | Learned gradient for x < 0 | ✅ Adaptive |
| **Swish** | Smooth gradient everywhere | ✅ Best for deep |

---

*📝 Notes compiled for NIELIT × IIT Ropar AI/ML Training Program — Deep Learning Module*
*🚀 Part of the AAgni AI Knowledge Base — Built in Patiala, Made in India 🇮🇳*