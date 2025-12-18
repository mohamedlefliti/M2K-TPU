# M2K-TPU

# M2K: Machine Learning Made Kingly Fast on TPUs

> A compact, educational framework for rapid prototyping of neural networks on TPUs using JAX — all in a single file.

M2K provides a high-level, PyTorch-like interface for defining models, training loops, and TPU setup, while staying lightweight and transparent. Designed for researchers and students who want to **learn how deep learning frameworks work** or **experiment quickly** without heavy dependencies.

✅ **Fully self-contained** (one file, no hidden modules)  
✅ **TPU-aware** with automatic device detection  
✅ Includes `Linear`, `ReLU`, `Dropout`, `Softmax`, `SGD`, `Adam`, and more  
✅ Built on **JAX** for functional, composable, and hardware-accelerated computation  

> 🔍 **Note**: M2K is a *minimal reference implementation* for educational and experimental purposes — not a production-grade library.

---

## Why M2K?

While tools like Flax, Haiku, or PyTorch are powerful, sometimes you need a **clear, readable abstraction** to:
- Understand the internals of training loops
- Prototype novel architectures quickly
- Run lightweight experiments on TPUs (e.g., in Google Colab)
- Teach or demonstrate ML concepts without abstraction overload

M2K strips away complexity while preserving core functionality — making the "magic" of deep learning more visible.

---

## Quick Start

```python
from m2k import init_tpu, Sequential, Linear, ReLU, Softmax, Trainer, CrossEntropyLoss, Adam

# Initialize hardware
init_tpu()

# Build model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
    Softmax()
)

# Dummy data
import jax.random as jr
X = jr.normal(jr.PRNGKey(0), (100, 784))
y = jr.randint(jr.PRNGKey(1), (100,), 0, 10)
y = jax.nn.one_hot(y, 10)

# Train
trainer = Trainer(model, Adam(model.parameters()), CrossEntropyLoss())
trainer.fit(X, y, epochs=5)
