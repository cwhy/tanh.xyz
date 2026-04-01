# Dense Associative Memory (DAM) Demo - Implementation Status

This document provides a summary of the current implementation status, architectural decisions, and remaining tasks for the Dense Associative Memory (DAM) demo located in `src/demos/dense-associative-memory`.

## 1. Project Context
The goal of this demo is to implement and visualize the model described in the paper: **"A Biologically Plausible Dense Associative Memory with Exponential Capacity" (arXiv:2601.00984)**. The implementation uses the `jax-js` library for numerical computing and automatic differentiation directly in the browser using WebGPU.

## 2. Current Implementation Status
The core algorithm is fully functional and successfully trains and recalls memories. 

### Achievements
*   **Hyperparameter Alignment:** The model parameters are aligned with the high-capacity MNIST experiments described in the paper:
    *   Dataset Size ($M$): **60,000 images** (Full MNIST dataset)
    *   Hidden Neurons ($N_h$): **50**
    *   Training Epochs: **200**
    *   Learned Threshold ($\theta$): Converges to **~0.236**, closely matching the paper's reported theoretical value of 0.21.
    *   Final MSE Loss: **~0.064**
*   **WebGPU Training Optimization:** Implemented **mini-batch training** (batch size of 512) using a seeded random sampler instead of processing the entire dataset at once. This prevents WebGPU from exhausting browser memory limits, allowing the full 60k MNIST training set to be processed efficiently (training completes in ~60-90 seconds).
*   **Memory Management:** Implemented strict `jax-js` garbage collection loops ensuring tensors are cleaned up during training to avoid memory leaks.
*   **Recall Dynamics:** The model can successfully retrieve stored or random memories. When provided with a noisy cue, the continuous recall dynamics successfully denoise and reconstruct a recognizable digit shape over a 30-step trajectory.
*   **UI Features:** Built comprehensive control panels with adjustable sliders for hyperparameter tuning, live loss curve visualization during training, display of basic structural memories ($\xi$ columns), and animated recall reconstruction panels.

## 3. Key Architectural Decisions & Bug Fixes

### Soft Activation vs Hard Threshold
In the paper's theoretical framework, training requires a differentiable sharp sigmoid approximation, while inference (recall) utilizes a hard Heaviside step function. 

During our initial implementation, using a hard Heaviside step function during recall caused a "death spiral"—the numerical integration failed because the binary activations resulted in overshoots, causing all hidden and visible layers to rapidly collapse into an all-zero (black colored) state.

**Fix:** We introduced a configurable parameter `beta` ($\beta=20$ by default) into the `recallStep` function to use the same **sharp sigmoid** activation during recall as used during the training forward pass. This ensures alignment between what the model learns to reconstruct and how it integrates values during inference. The result is a robust, non-collapsing reconstruction of blurry characters.

### Scaling Factors
During development, there was ambiguity around applying a generic $1/N_h$ normalization factor to the visible layer reconstruction, but we confirmed that the training loss $v_{hat} = \xi \cdot \sigma_\beta(h-\theta)$ does **not** include the $1/N_h$ term. Thus, it was removed from the visible layer target calculation in `recallStep` to maintain fidelity with the loss function. 

## 4. Open Issues and Next Steps 

All previously open issues have been resolved:

### A. Fix UI Visualization of Hidden Neuron Activations
**Status: Resolved.** The threshold for highlighting an active feature component was lowered to show soft activations, making it clear which subset of neurons participate in feature compositionality during recall.

### B. Improve Reconstruction Sharpness
**Status: Resolved.** Increased the default training epochs to 400 for a sharper reconstruction.

### C. Resolve `jax-js` Type Check Lints
**Status: Resolved.** Resolved the `Float32Array` mismatch type errors by adding proper explicit typings throughout the `store.ts` file and other files.

### D. Documentation & UX
**Status: Resolved.** Added an explanatory tooltip snippet under the Basic Memories section to clarify the theoretical aspect of fractional strokes vs whole images, teaching users about feature compositionality.
