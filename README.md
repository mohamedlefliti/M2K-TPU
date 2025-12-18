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
this resulats of prototype : 
<img width="733" height="590" alt="téléchargement (73)" src="https://github.com/user-attachments/assets/8c8df736-1bdc-47c2-bd38-c60c06188104" />

<img width="1189" height="390" alt="téléchargement (72)" src="https://github.com/user-attachments/assets/9096ccbd-8e97-42e0-b67d-2ecaeac0ed81" />

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


# final output :
============================================================
📸 IMAGE CLASSIFICATION WITH M2K LIBRARY
============================================================
✅ JAX version: 0.7.2

============================================================
🏁 M2K IMAGE CLASSIFICATION DEMO
============================================================

🎯 Running Image Classifier Training...

============================================================
🤖 TRAINING IMAGE CLASSIFIER WITH M2K
============================================================

============================================================
🖼️ Creating synthetic image classification dataset
============================================================
✅ Dataset created successfully!
   Total samples: 800
   Features per sample: 100
   Number of classes: 5
   Training samples: 560
   Validation samples: 80
   Test samples: 160

============================================================
🏗️ Building Neural Network Model
============================================================

Module: SimpleClassifier
  SimpleClassifier.linear_0.linear_0.weight: (100, 50) (5,000)
  SimpleClassifier.linear_0.linear_0.bias: (50,) (50)
  SimpleClassifier.linear_2.linear_2.weight: (50, 5) (250)
  SimpleClassifier.linear_2.linear_2.bias: (5,) (5)
Total parameters: 5,305

============================================================
🚂 Training Model
============================================================

🚀 Starting training for 15 epochs
   Batch size: 32
   Learning rate: 0.01
   Training samples: 560
   Validation samples: 80

--------------------------------------------------------------------------------
Epoch  | Train Loss | Train Acc | Val Loss | Val Acc | Time
--------------------------------------------------------------------------------
    1 |     1.9456 |    0.2170 |   1.8554 |  0.2125 |  0.40s
    5 |     1.6025 |    0.3021 |   1.6691 |  0.2125 |  1.03s
   10 |     1.4027 |    0.4184 |   1.5861 |  0.3125 |  1.37s
   15 |     1.2614 |    0.5087 |   1.5361 |  0.3375 |  0.70s
--------------------------------------------------------------------------------

✅ Training completed!

============================================================
📈 Training History
============================================================


============================================================
🧪 Final Evaluation
============================================================

🧪 Evaluating on test set...
📊 Test Evaluation:
   Loss: 1.5429
   Accuracy: 0.3750 (37.50%)

📊 Test Set Performance:
   Accuracy: 0.3750 (37.50%)

📋 Classification Report:
              precision    recall  f1-score   support

           0       0.29      0.29      0.29        21
           1       0.42      0.42      0.42        38
           2       0.33      0.34      0.34        35
           3       0.44      0.32      0.38        37
           4       0.37      0.48      0.42        29

    accuracy                           0.38       160
   macro avg       0.37      0.37      0.37       160
weighted avg       0.38      0.38      0.37       160



============================================================
🔮 Sample Predictions
============================================================

Sample predictions:
----------------------------------------
 Index     True     Pred     Status
----------------------------------------
     0        3        0          ✗
     1        0        1          ✗
     2        0        0          ✓
     3        2        3          ✗
     4        1        2          ✗
----------------------------------------
Accuracy on 5 samples: 1/5 = 20.00%

============================================================
💾 Model Information
============================================================
📊 Final Statistics:
   Model: SimpleClassifier
   Final training accuracy: 0.5087 (50.87%)
   Final validation accuracy: 0.3375 (33.75%)
   Test accuracy: 0.3750 (37.50%)
   Total training time: 12.57 seconds

============================================================
🎉 DEMO COMPLETED SUCCESSFULLY!
============================================================

📋 Summary:
✅ Neural network library (M2K) is working!
✅ Image classification pipeline executed successfully!
✅ Model trained and evaluated!
✅ All visualizations generated!

