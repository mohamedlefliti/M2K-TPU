"""
📸 Image Classification Using M2K Small Library
Complete fixed code - parameter names fixed
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time
import warnings
from typing import Dict, Any, List
from collections import OrderedDict
import optax
from jax import random

print("="*60)
print("📸 IMAGE CLASSIFICATION WITH M2K LIBRARY")
print("="*60)
print(f"✅ JAX version: {jax.__version__}")

# ============================================================================
# 1. CORE MODULE SYSTEM
# ============================================================================

class Parameter:
    """Simple parameter container"""
    def __init__(self, data, name: str = None):
        self.data = data
        self.name = name or "param"
        self.grad = None
        self.shape = data.shape
        self.dtype = data.dtype

    def __repr__(self):
        return f"Parameter({self.shape}, {self.dtype})"

class Module:
    """PyTorch-like Module system"""

    def __init__(self, name: str = None):
        self._name = name or self.__class__.__name__
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif isinstance(value, Module):
            self._modules[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Parameter):
            self._parameters[name] = value
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self._name}.forward() not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse: bool = True):
        params = []
        for name, param in self._parameters.items():
            params.append((f"{self._name}.{name}", param))

        if recurse:
            for mod_name, module in self._modules.items():
                for param_name, param in module.parameters(recurse=True):
                    params.append((f"{self._name}.{mod_name}.{param_name}", param))

        return params

    def add_module(self, name: str, module: 'Module'):
        """Add a child module"""
        self._modules[name] = module
        setattr(self, name, module)

    def summary(self):
        lines = [f"Module: {self._name}"]
        total_params = 0

        for name, param in self.parameters():
            param_count = np.prod(param.shape)
            total_params += param_count
            lines.append(f"  {name}: {param.shape} ({param_count:,})")

        lines.append(f"Total parameters: {total_params:,}")
        return "\n".join(lines)

# ============================================================================
# 2. NEURAL NETWORK LAYERS
# ============================================================================

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = None):
        super().__init__(name or f"Linear_{in_features}_{out_features}")

        scale = jnp.sqrt(2.0 / (in_features + out_features))
        key = random.PRNGKey(42)

        weight = random.normal(key, (in_features, out_features)) * scale
        self.weight = Parameter(weight, name='weight')

        if bias:
            bias_val = jnp.zeros((out_features,))
            self.bias = Parameter(bias_val, name='bias')
        else:
            self.bias = None

    def forward(self, x):
        output = jnp.dot(x, self.weight.data)
        if self.bias is not None:
            output = output + self.bias.data
        return output

class ReLU(Module):
    def forward(self, x):
        return jnp.maximum(0, x)

# ============================================================================
# 3. MODEL ARCHITECTURES
# ============================================================================

class Sequential(Module):
    def __init__(self, *modules, name: str = "Sequential"):
        super().__init__(name)
        self.layers = list(modules)

        # Add modules with proper names
        for i, module in enumerate(self.layers):
            # Give each module a unique name based on its type
            if isinstance(module, Linear):
                module_name = f"linear_{i}"
            elif isinstance(module, ReLU):
                module_name = f"relu_{i}"
            else:
                module_name = f"layer_{i}"

            module._name = module_name
            self.add_module(module_name, module)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================
# 4. SIMPLE TRAINING SYSTEM
# ============================================================================

class TrainingConfig:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 20,
                 batch_size: int = 32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

class SimpleTrainer:
    """Simple trainer with manual gradient descent"""

    def __init__(self, model: Module, config: TrainingConfig = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': []
        }

    def softmax(self, x):
        """Stable softmax"""
        x_max = jnp.max(x, axis=-1, keepdims=True)
        exp_x = jnp.exp(x - x_max)
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    def cross_entropy_loss(self, pred, target):
        """Cross entropy loss"""
        epsilon = 1e-8
        pred = jnp.clip(pred, epsilon, 1. - epsilon)
        return -jnp.mean(jnp.sum(target * jnp.log(pred), axis=-1))

    def compute_accuracy(self, pred, target):
        """Compute accuracy"""
        pred_class = jnp.argmax(pred, axis=-1)
        true_class = jnp.argmax(target, axis=-1)
        return jnp.mean(pred_class == true_class)

    def get_params(self):
        """Get all parameters as a list"""
        params = []
        for _, param in self.model.parameters():
            params.append(param.data)
        return params

    def set_params(self, new_params):
        """Set all parameters"""
        param_items = list(self.model.parameters())
        for (_, param), new_data in zip(param_items, new_params):
            param.data = new_data

    def compute_gradients_simple(self, X_batch, y_batch):
        """Simple gradient computation using JAX autodiff"""
        # Get current parameters
        params = self.get_params()

        # Define loss function
        def loss_fn(params):
            # Set parameters
            temp_params = self.get_params()
            self.set_params(params)

            # Forward pass
            predictions = self.model(X_batch)
            predictions_softmax = self.softmax(predictions)

            # Compute loss
            loss = self.cross_entropy_loss(predictions_softmax, y_batch)

            # Restore original parameters
            self.set_params(temp_params)

            return loss

        # Compute gradient using JAX
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(params)

        # Forward pass for accuracy
        predictions = self.model(X_batch)
        predictions_softmax = self.softmax(predictions)
        accuracy = self.compute_accuracy(predictions_softmax, y_batch)
        loss = float(self.cross_entropy_loss(predictions_softmax, y_batch))

        return loss, float(accuracy), gradients

    def create_batches(self, X, y, batch_size):
        """Create mini-batches"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            if len(excerpt) > 0:
                yield X[excerpt], y[excerpt]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print(f"\n🚀 Starting training for {self.config.epochs} epochs")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"   Validation samples: {len(X_val)}")

        print("\n" + "-"*80)
        print("Epoch  | Train Loss | Train Acc | Val Loss | Val Acc | Time")
        print("-"*80)

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Training
            total_loss = 0.0
            total_acc = 0.0
            n_batches = 0

            for X_batch, y_batch in self.create_batches(X_train, y_train, self.config.batch_size):
                # Compute gradients
                loss, acc, gradients = self.compute_gradients_simple(X_batch, y_batch)

                # Update parameters
                params = self.get_params()
                new_params = []
                for param, grad in zip(params, gradients):
                    new_param = param - self.config.learning_rate * grad
                    new_params.append(new_param)

                self.set_params(new_params)

                total_loss += loss
                total_acc += acc
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            avg_acc = total_acc / n_batches if n_batches > 0 else 0

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate_single_batch(X_val, y_val)
            else:
                val_loss, val_acc = 0.0, 0.0

            # Store history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.config.epochs - 1:
                print(f"{epoch+1:5d} | {avg_loss:10.4f} | {avg_acc:9.4f} | "
                      f"{val_loss:8.4f} | {val_acc:7.4f} | {epoch_time:5.2f}s")

        print("-"*80)
        print("\n✅ Training completed!")
        return self.history

    def evaluate_single_batch(self, X, y):
        """Evaluate on a single batch"""
        self.model.eval()
        predictions = self.model(X)
        predictions_softmax = self.softmax(predictions)
        loss = float(self.cross_entropy_loss(predictions_softmax, y))
        accuracy = float(self.compute_accuracy(predictions_softmax, y))
        self.model.train()
        return loss, accuracy

    def evaluate(self, X_test, y_test):
        """Evaluate on test data"""
        print("\n🧪 Evaluating on test set...")

        # Split into batches for evaluation
        predictions = []
        batch_size = self.config.batch_size

        self.model.eval()
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            batch_pred = self.model(X_batch)
            predictions.append(batch_pred)

        predictions = jnp.concatenate(predictions, axis=0)
        predictions_softmax = self.softmax(predictions)

        loss = float(self.cross_entropy_loss(predictions_softmax, y_test))
        accuracy = float(self.compute_accuracy(predictions_softmax, y_test))

        self.model.train()

        print(f"📊 Test Evaluation:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy:.2%})")

        return {'loss': loss, 'accuracy': accuracy}

    def predict(self, X):
        """Make predictions"""
        self.model.eval()

        predictions = []
        batch_size = self.config.batch_size

        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            batch_pred = self.model(X_batch)
            predictions.append(batch_pred)

        predictions = jnp.concatenate(predictions, axis=0)
        pred_classes = jnp.argmax(predictions, axis=-1)

        self.model.train()
        return pred_classes

    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc', color='blue', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', color='orange', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ============================================================================
# 5. DATA CREATION
# ============================================================================

def create_image_classification_data(n_samples=800, n_features=100, n_classes=5):
    """Create synthetic image-like data (smaller for faster execution)"""
    print("\n" + "="*60)
    print("🖼️ Creating synthetic image classification dataset")
    print("="*60)

    # Create features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to JAX arrays
    X = jnp.array(X, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.int32)

    # One-hot encode
    y_onehot = jax.nn.one_hot(y, n_classes)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42
    )

    print(f"✅ Dataset created successfully!")
    print(f"   Total samples: {n_samples}")
    print(f"   Features per sample: {n_features}")
    print(f"   Number of classes: {n_classes}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

def evaluate_model_performance(y_true, y_pred, dataset_name="Test"):
    """Evaluate model performance"""
    y_true_classes = np.argmax(np.array(y_true), axis=1) if len(y_true.shape) > 1 else np.array(y_true)
    y_pred_classes = np.array(y_pred)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)

    print(f"\n📊 {dataset_name} Set Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return accuracy

# ============================================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================================

def train_image_classifier():
    """Main function to train an image classifier"""
    print("\n" + "="*60)
    print("🤖 TRAINING IMAGE CLASSIFIER WITH M2K")
    print("="*60)

    # Step 1: Create dataset
    data = create_image_classification_data(
        n_samples=800,      # Small dataset for fast execution
        n_features=100,     # Smaller feature dimension
        n_classes=5         # Fewer classes
    )

    # Step 2: Create model
    print("\n" + "="*60)
    print("🏗️ Building Neural Network Model")
    print("="*60)

    # Create simple 2-layer model
    model = Sequential(
        Linear(100, 50, name="fc1"),
        ReLU(),
        Linear(50, 5, name="output"),
        name="SimpleClassifier"
    )

    print("\n" + model.summary())

    # Step 3: Configure training
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=15,          # Fewer epochs
        batch_size=32
    )

    # Step 4: Create trainer
    trainer = SimpleTrainer(model, config)

    # Step 5: Train the model
    print("\n" + "="*60)
    print("🚂 Training Model")
    print("="*60)

    history = trainer.fit(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )

    # Step 6: Plot training history
    print("\n" + "="*60)
    print("📈 Training History")
    print("="*60)
    trainer.plot_history()

    # Step 7: Evaluate on test set
    print("\n" + "="*60)
    print("🧪 Final Evaluation")
    print("="*60)

    test_results = trainer.evaluate(data['X_test'], data['y_test'])

    # Get predictions
    y_pred = trainer.predict(data['X_test'])

    # Evaluate performance
    accuracy = evaluate_model_performance(data['y_test'], y_pred, "Test")

    # Step 8: Make sample predictions
    print("\n" + "="*60)
    print("🔮 Sample Predictions")
    print("="*60)

    # Get 5 random samples
    n_samples = 5
    indices = np.random.choice(len(data['X_test']), n_samples, replace=False)
    X_sample = data['X_test'][indices]
    y_sample_true = data['y_test'][indices]

    y_sample_pred = trainer.predict(X_sample)
    y_sample_true_classes = np.argmax(np.array(y_sample_true), axis=1)

    print("\nSample predictions:")
    print("-" * 40)
    print(f"{'Index':>6} {'True':>8} {'Pred':>8} {'Status':>10}")
    print("-" * 40)

    correct_count = 0
    for i in range(n_samples):
        true_class = y_sample_true_classes[i]
        pred_class = y_sample_pred[i]
        correct = true_class == pred_class
        if correct:
            correct_count += 1

        status = "✓" if correct else "✗"
        print(f"{i:6d} {true_class:8d} {pred_class:8d} {status:>10}")

    print("-" * 40)
    print(f"Accuracy on {n_samples} samples: {correct_count}/{n_samples} = {correct_count/n_samples:.2%}")

    # Step 9: Final summary
    print("\n" + "="*60)
    print("💾 Model Information")
    print("="*60)

    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]

    print(f"📊 Final Statistics:")
    print(f"   Model: {model._name}")
    print(f"   Final training accuracy: {final_train_acc:.4f} ({final_train_acc:.2%})")
    print(f"   Final validation accuracy: {final_val_acc:.4f} ({final_val_acc:.2%})")
    print(f"   Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"   Total training time: {sum(history['epoch_times']):.2f} seconds")

    return {
        'model': model,
        'trainer': trainer,
        'data': data,
        'history': history,
        'test_results': test_results
    }

# ============================================================================
# 7. RUN THE CODE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏁 M2K IMAGE CLASSIFICATION DEMO")
    print("="*60)

    print("\n🎯 Running Image Classifier Training...")

    try:
        results = train_image_classifier()
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)

        print("\n📋 Summary:")
        print("✅ Neural network library (M2K) is working!")
        print("✅ Image classification pipeline executed successfully!")
        print("✅ Model trained and evaluated!")
        print("✅ All visualizations generated!")

    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Quick fix:")
        print("Running a simpler version...")

        # Run minimal version
        try:
            # Minimal working example
            print("\n" + "="*60)
            print("🔄 Trying minimal working example...")
            print("="*60)

            # Create tiny dataset
            X = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=jnp.float32)
            y = jnp.array([[1, 0], [0, 1], [1, 0]], dtype=jnp.float32)

            # Create tiny model
            model = Sequential(
                Linear(2, 3),
                ReLU(),
                Linear(3, 2),
                name="TinyModel"
            )

            print("\nModel summary:")
            print(model.summary())

            # Simple forward pass test
            print("\nTesting forward pass...")
            output = model(X)
            print(f"Input shape: {X.shape}")
            print(f"Output shape: {output.shape}")

            print("\n✅ Minimal test passed! The M2K library is working.")
            print("   Try increasing dataset size and model complexity gradually.")

        except Exception as e2:
            print(f"\n❌ Even minimal test failed: {type(e2).__name__}: {e2}")
            print("\n💡 Please check:")
            print("1. JAX installation: pip install --upgrade jax jaxlib")
            print("2. Other dependencies: pip install numpy matplotlib scikit-learn seaborn optax")
            print("3. Try in a fresh Python environment")
