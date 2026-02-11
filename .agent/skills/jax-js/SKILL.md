---
name: jax-js
description: Guide for using the jax-js library (JAX port for JavaScript/TypeScript) for numerical computing, automatic differentiation, and neural network training in the browser.
---

# jax-js Library Guide

`jax-js` is a JavaScript/TypeScript port of Google's JAX library. It runs entirely in the browser using WebGPU (preferred) or WASM and provides NumPy-like array operations, automatic differentiation, JIT compilation, and an optimizer library (`@jax-js/optax`).

**Packages:**
- `@jax-js/jax` — Core: arrays, numpy ops, autodiff (`grad`, `valueAndGrad`), `jit`, `vmap`, tree utilities
- `@jax-js/optax` — Optimizers: `adam`, `sgd`, `adamw`, loss functions, gradient transformations

## Initialization

Always call `init()` before using any array operations. It returns available devices.

```ts
import { init, defaultDevice } from '@jax-js/jax'

const devices = await init()
if (devices.includes('webgpu')) {
    defaultDevice('webgpu')
}
```

Available devices: `'cpu'`, `'wasm'`, `'webgpu'`, `'webgl'`.

## Core Imports

```ts
import { numpy as np, tree, init, defaultDevice, valueAndGrad, grad, jit, vmap } from '@jax-js/jax'
import { adam, sgd, adamw, applyUpdates, type OptState } from '@jax-js/optax'
```

Note: `numpy` is aliased as `np` by convention (mirrors Python's `import numpy as np`).

## Array Creation

```ts
const a = np.array([1, 2, 3])           // from JS array
const b = np.array([[1, 2], [3, 4]])     // 2D
const z = np.zeros([3, 3])              // zeros
const o = np.ones([2, 4])               // ones
const r = np.linspace(0, 1, 50)         // 50 evenly spaced values
const e = np.eye(3)                     // identity matrix
const rng = np.arange(0, 10, 2)         // [0, 2, 4, 6, 8]
```

## ⚠️ Memory Management — Move Semantics (CRITICAL)

**jax-js uses Rust/C++-style move semantics.** When you pass an array into a function/operation, it is **consumed** (reference count decremented). You cannot reuse it after.

```ts
// ❌ WRONG — x is consumed by the first argument
const y = np.add(x, x)

// ✅ CORRECT — use .ref to increment reference count
const y = np.add(x.ref, x)

// ✅ CORRECT — Method chaining also consumes
const y = x.ref.add(x)
```

### Rules:
1. Every array starts with refcount = 1
2. Passing to an operation decrements refcount (consumes the array)
3. `.ref` increments the refcount (creates a non-owning reference)
4. When refcount hits 0, the GPU/CPU memory is freed
5. For tree structures (param objects), use `tree.ref(params)` and `tree.dispose(params)`

### When to use `.ref`:
- When the **same array** is used **multiple times** in an expression
- When you want to keep an array alive after passing it to a function
- When passing params to `valueAndGrad` — the params are consumed, so use `tree.ref(params)`

### When to use `.dispose()`:
- Manually free arrays you no longer need (e.g., old training data)
- When a function receives an argument it doesn't use, convention is to dispose it
- For tree structures: `tree.dispose(params)` disposes all arrays in the tree

## Extracting Values — `.js()` vs `.jsAsync()`

To get JavaScript values out of a jax-js array:

```ts
// ❌ BLOCKING — freezes the UI during shader compilation + execution
const result = preds.js() as number[][]

// ✅ NON-BLOCKING — allows browser to remain responsive
const result = await preds.jsAsync() as number[][]
```

**Always prefer `.jsAsync()` in production code.** The synchronous `.js()` blocks the entire main thread while waiting for WebGPU shader compilation and execution. `.jsAsync()` returns a `Promise` that resolves when the GPU work is done.

Other data access methods:
- `.data()` — async, returns raw `DataArray` (typed array)
- `.dataSync()` — sync, returns raw `DataArray` (not recommended)
- `.item()` — for scalars only, returns a `number`

## Array Operations

Method chaining style (preferred):
```ts
const h = np.tanh(np.matmul(X, W).add(b))        // tanh(X @ W + b)
const logits = np.matmul(h, W2).add(b2)
const sigmoid = np.reciprocal(np.exp(x.neg()).add(1))  // 1/(1+exp(-x))
```

Common operations (both function and method form):
```ts
// Arithmetic
a.add(b)    a.sub(b)    a.mul(b)    a.div(b)    a.neg()

// Comparison
a.greater(b)  a.less(b)   a.equal(b)  a.greaterEqual(b)

// Reductions
a.sum()     a.mean()    a.max()     a.min()     a.prod()
a.sum(0)    a.mean(1)   // along axis

// Shape
a.reshape([2, 3])   a.transpose()   a.flatten()
a.slice([1, 3], 2)  // slicing

// Math functions (np. prefix)
np.exp(x)   np.log(x)   np.tanh(x)   np.sqrt(x)
np.abs(x)   np.sin(x)   np.cos(x)
np.matmul(a, b)   np.dot(a, b)

// Linear algebra (np.linalg.)
np.linalg.inv(a)     np.linalg.det(a)
np.linalg.solve(a, b)   np.linalg.cholesky(a)
```

## Automatic Differentiation

### `grad(f)` — Gradient of scalar-output function
```ts
const f = (x: np.Array) => x.mul(x.ref).sum()  // sum of squares
const df = grad(f)
const gradient = df(np.array([1, 2, 3]))
```

### `valueAndGrad(f)` — Value AND gradient in one pass
```ts
function loss(params: Params, X: np.Array, y: np.Array): np.Array {
    const preds = forward(params, X)
    return computeLoss(preds, y)
}

// @ts-expect-error — P satisfies MappedJsTree<P, Array, ArrayLike> but TS can't verify conditional types on generics
const [lossVal, grads] = valueAndGrad(loss)(tree.ref(params), X, y) as [np.Array, Params]

const lossNumber = await lossVal.jsAsync() as number
```

### Options
```ts
// Differentiate w.r.t. specific arguments
grad(f, { argnums: 0 })        // default: first arg
grad(f, { argnums: [0, 1] })   // multiple args

// Auxiliary return values (not differentiated)
const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x)
```

## JIT Compilation

```ts
const fastFn = jit((x: np.Array, y: np.Array) => x.add(y).mul(x))
const result = fastFn(a, b)

// With options
const fastFn = jit(myFn, { name: 'myKernel' })
```

`jit` returns an `OwnedFunction` — these compile on first call and cache the shader.

## Vectorized Map (`vmap`)

```ts
const batchedFn = vmap(singleExampleFn)
const results = batchedFn(batchOfInputs)
```

## Optimizers (`@jax-js/optax`)

```ts
import { adam, sgd, adamw, applyUpdates, type OptState } from '@jax-js/optax'

// Create optimizer
const solver = adam(0.001)   // or sgd(0.01), adamw(0.001)

// Initialize state
const optState: OptState = solver.init(tree.ref(params))

// Update step
const [updates, newOptState] = solver.update(grads, optState)
const newParams = applyUpdates(params, updates)
```

### Full training step pattern
```ts
async function trainStep(params, optState, X, y, learningRate) {
    const solver = adam(learningRate)

    const [lossVal, grads] = valueAndGrad(lossFn)(
        tree.ref(params) as any, X, y
    ) as [np.Array, typeof params]

    const loss = await lossVal.jsAsync() as number

    const [updates, newOptState] = solver.update(grads, optState)
    const newParams = applyUpdates(params, updates)

    return { params: newParams, optState: newOptState, loss }
}
```

### Available loss functions
```ts
import { squaredError, l2Loss } from '@jax-js/optax'

squaredError(predictions, targets)       // element-wise (pred - target)^2
l2Loss(predictions, targets)             // 0.5 * squared_error
```

## Tree Utilities (`tree`)

Trees (JsTree) are nested structures of arrays — objects, arrays of arrays, etc. Used for model parameters.

```ts
import { tree } from '@jax-js/jax'

tree.ref(params)      // increment refcount for all arrays in tree
tree.dispose(params)  // dispose all arrays in tree
tree.map(fn, params)  // apply fn to each leaf array
tree.leaves(params)   // get flat list of all arrays
tree.flatten(params)  // [leaves[], treeDef]
```

## TypeScript Gotchas

### 1. `valueAndGrad` generic type mismatch
When using `valueAndGrad` with generic param types, you'll hit:
```
Argument of type 'P' is not assignable to parameter of type 'MappedJsTree<P, Array, ArrayLike>'
```

**Fix:** Use `@ts-expect-error` on the line before the call (collapsed to one line so the directive applies):
```ts
// @ts-expect-error — P satisfies MappedJsTree<P, Array, ArrayLike> but TS can't verify conditional types on generics
const [lossVal, grads] = valueAndGrad(lossFn)(tree.ref(params), X, y) as [np.Array, P]
```

This is safe because `Array` extends `ArrayLike`. The issue is that TypeScript can't verify the `MappedJsTree` conditional type mapping for generic `P`.

### 2. Return type of `.js()` / `.jsAsync()`
Both return `any`. Always cast the result:
```ts
const values = await arr.jsAsync() as number[][]
const scalar = await arr.jsAsync() as number
```

### 3. `np.Array` conflicts with JavaScript `Array`
Since jax-js exports a class named `Array`, avoid importing it directly into the namespace. Use the `numpy as np` pattern:
```ts
// ✅ Good
import { numpy as np } from '@jax-js/jax'
type Params = { W: np.Array; b: np.Array }

// ❌ Conflicts with JS Array
import { Array } from '@jax-js/jax'
```

## Common Patterns in This Project

### Parameter definition
```ts
interface NetworkParams {
    W1: np.Array
    b1: np.Array
    W2: np.Array
    b2: np.Array
    [key: string]: np.Array  // Index signature for JsTree compatibility
}
```

### He initialization
```ts
function initParams(hiddenSize: number): NetworkParams {
    const scale = Math.sqrt(2 / inputDim)
    return {
        W: np.array(Array.from({ length: inputDim }, () =>
            Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * 2 * scale)
        )),
        b: np.zeros([hiddenSize]),
    }
}
```

### Async training loop with UI updates
```ts
async function runTraining() {
    for (let i = 0; i < epochs; i++) {
        const result = await trainStep(params, optState, X, y, lr)
        params = result.params
        optState = result.optState

        // Allow UI to update between epochs
        await new Promise(r => setTimeout(r, 0))
    }
}
```

### Batch prediction with async
```ts
async function predictBatch(params: NetworkParams, X: np.Array): Promise<number[]> {
    const preds = forward(tree.ref(params), X.ref)
    const result = await preds.jsAsync() as number[][]
    return result.flat()
}
```

## File Locations in This Project

- **Shared deep learning utilities:** `src/lib/deep-learning/deep-learning.ts`
  - `initJax()`, `initOptimizer()`, `trainStep()` — generic, reusable
- **Two moons demo:** `src/demos/two-moons-clustering/`
  - `models.ts` — network architecture, forward pass, loss, prediction
  - `training-store.ts` — SolidJS reactive training state management
  - `TwoMoonDemo.tsx` — UI component
