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

Please address the following items to finalize the demo:

### A. Fix UI Visualization of Hidden Neuron Activations
**Issue:** During the recall process, the reconstruction succeeds, but the UI displays **"0/50 active"** hidden neurons, and all 50 neuron progress bars remain completely empty.
**Context:** We modified `RecallFrame` to store both `hBinary` (the hard threshold) and `hActivated` (the sigmoid soft activation). The rendering logic in `DenseAssociativeMemoryDemo.tsx` processes the component `HiddenBar`. Currently, `isActive={ha() > 0.5}` expects activations to cross 0.5 in order to switch to green or be counted. During our tests with the $\beta=20$ sigmoid, the actual driving activations $h_\mu$ fall below `0.5`, causing the UI to mask their contribution despite them collectively forming the visible reconstruction.
**Next Step:** Adjust the visualization mapping. Either visualize the continuous scalar value directly using opacity/color scaling without a rigid binary cutoff, or lower the strict threshold for visualization so that users can actually see which features the network is utilizing during recall.

### B. Improve Reconstruction Sharpness
**Issue:** While the reconstructed digits are recognizable and stable, they appear slightly blurry/averaged. 
**Next Step:** Consider experimenting with higher training epochs, tweaking the learning rate scheduling, or adjusting the $\beta$ activation curve. Although blurry recall is partially a feature of Dense Associative Memories overlapping features in continuous state spaces, you might be able to find a cleaner convergence hyper-plane.

### C. Resolve `jax-js` Type Check Lints
**Issue:** There are residual TypeScript lint errors in `store.ts` pointing to: `Type 'Float32Array<ArrayBufferLike>' is not assignable to type 'Float32Array<ArrayBuffer>'`. 
**Next Step:** These are caused by standard `jax`.js return types using `ArrayBufferLike` due to WebGPU/SharedArrayBuffer complexities. It doesn't break browser runtime. You may ignore them with standard `@ts-expect-error` or fix the types in your structural definitions.

### D. Documentation & UX
**Next Step:** Consider adding a "Theory Snippet" or tooltip text helping users understand *why* the basic memory weights look like fractional strokes of digits rather than full digits (which highlights the paper's core assertion about exponential capacity and feature compositionality vs standard Hopfield nets).
