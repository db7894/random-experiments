# Running passes in this folder

## Setup notes
Jeremy Kun has a very helpful guide on setting up LLVM that I used to create the initial `WORKSPACE` and `bazel/` setup: https://www.jeremykun.com/2023/08/10/mlir-getting-started/

Also generally a good idea to set up a Bazel cache somewhere so you're not building LLVM / dependencies from scratch every time you want to run this. 

## Loop Detection Pass
This pass basically does nothing. It just checks if you have a loop from the `scf` dialect and spits it back out if so. 
You can run it with the following two commands, using the `loop_example.mlir` in the `tests/` directory or using your own example. 
```
bazel build //:optimizer
bazel-bin/optimizer tests/loop_example.mlir -o output.mlir
```
With the current example, you should expect an output like this:
```
Found a loop in function simple_loop: scf.for %arg4 = %c0 to %arg1 step %c1 {
  %0 = arith.addi %arg3, %arg4 : index
  %1 = arith.muli %0, %arg2 : index
}
```

## Loop Invariant Code Motion
This pass performs simple loop-invariant code motion using the `scf` dialect. I've written simple functionality to get dominance information for ops inside (nested) `for` ops and determine their loop invariance. 

### Computing Dominators
You can find standard algorithms for dominance and loop invariance in sources like [this](https://piazza.com/class_profile/get_resource/ixkj4uoml3y6qd/izeb0ekl6gsn8). 

The algorithm takes a few steps:
1. Perform a postorder traversal of the CFG. This is implemented basically as a DFS. 
2. Initialize the dominator set for each node to be _all_ nodes, except for the root node which dominates only itself. 
3. Iterate over nodes in reverse post-order, and for each:
  a. Intersect the dominator sets of all predecessors (node A dominates node B if A dominates all of B's predecessors)
  b. Add the node itself to the intersection (nodes dominate themselves)
  c. Update the node's current dominator set if the intersection is different (do-while; we're basically looking for a fixed point). 
4. Repeat (3) until we reach a fixed point

### Computing Loop Invariants
Loop-invariant operations produce the same result in every loop iteration. Formally, there are two criteria for this (+ a technical bonus):
- All of its operands are defined outside the loop
- All of its operands are themselves loop-invariant (recursive part of the definition)
- (bonus) The operation has no side effects (since e.g. printing inside a loop will occur multiple times)

So, to check these criteria, we use the dominator information we computed:
- For each op in the loop body, check that all of its operands' defining ops dominate the loop header (this means those operands were defined before the loop, so it's safe to hoist this op from the loop body).

### Running the pass
You can run the pass with similar bazel build commands as above. 
```
bazel build //:optimizer
bazel-bin/optimizer tests/licm_example.mlir -p loop-optimization -o licm_out.mlir
```

You should find that the two operations I've given leading names
```
%loop_invariant = arith.muli %inv0, %inv1 : i32
%loop_invariant2 = arith.addi %inv0, %inv1 : i32
```
are hoisted before the loop in your output IR. 

### Debug Info
You'll notice a bunch of debugging statements I've littered throughout the pass. They won't print on an ordinary run of the binary, but if you add a flag:
```
bazel-bin/optimizer --debug-pass-opt tests/licm_example.mlir -o licm_out.mlir
```
you'll see the debug logging. (I'd have just named it `debug`, but I LLVM already has its own `debug` flags I was probably importing somewhere so this caused re-definition errors)

## Loop Unrolling

### Notes on the Pass
Compilers generally perform loop unrolling to increase the # of instructions executed between executions of the loop branch logic. [This Stack Overflow post](https://stackoverflow.com/questions/2349211/when-if-ever-is-loop-unrolling-still-useful) has some notes. Unrolled loops can be vectorized, exploit cache locality, etc. 

The tl;dr is that you aim to gain some execution speed (by parallelizing statements in the loop if they're not dependent on one another) at the expense of instruction count / binary size. [This article](https://lemire.me/blog/2019/11/12/unrolling-your-loops-can-improve-branch-prediction/) explains how unrolling loops can improve branch prediction. 

[ TODO : add some concrete examples from Godbolt ]

N/B: I chose an arbitrary unroll factor (# times to replicate the loop body) of 4. Haven't gotten around to something more dynamic yet. 

#### Determining whether to unroll
To determine if the loop should be unrolled:
- we check if loop bounds and step are constant: if the step isn't constnat we don't unroll.
- if upper and lower bound are constant, we calculate the _trip count_ (# loop iterations) and unroll if it's greater than 1.
- if upper bound is not constant, we also allow unrolling w/ a runtime check
  - this is `remainingIters`, `unrollThreshold`, and `unrollCondition`. 

The `unrollLoop` method does the actual work: `newForOp` is to replace the original `forOp`. It has same bounds/step but contains unrolled code, and an if-else structure that unrolls in the `if` block (when we can unroll) and just clones the loop body in the `else` block (when we can't). 

#### Performing unrolling
This code from the `if` block:
```
mlir::Value currentIV = iv;
mlir::ValueRange currentIterArgs = iterArgs;
mlir::SmallVector<mlir::Value, 6> finalYieldOperands;

for (int i = 0; i < unrollFactor; ++i) {
  mlir::IRMapping mapping;
  mapping.map(forOp.getInductionVar(), currentIV);
  mapping.map(forOp.getRegionIterArgs(), currentIterArgs);

  for (mlir::Operation &op :
        forOp.getBody()->without_terminator()) {
    thenBuilder.clone(op, mapping);
  }
```
replicates the loop body `unrollFactor` times. The mapping stores what the current iv and block arguments (e.g. a value we're accumulating into during a loop) are. 

We need this mapping when we clone operations from the loop body with `thenBuilder.clone(op, mapping);` to make sure we're using the correct arguments to that op. Internally, if we look at MLIR's `Operation.h` (for docstring) and `Operation.cpp` for what `op.clone(mapping)` does, we see how remapping of operands happens during cloning (as of writing, this is line 682 of `mlir/lib/IR/Operation.cpp`):
```
// Remap the operands.
if (options.shouldCloneOperands()) {
  operands.reserve(getNumOperands());
  for (auto opValue : getOperands())
    operands.push_back(mapper.lookupOrDefault(opValue));
}
```

To make this a bit more concrete, compare the original IR example I use:
```
func.func @unroll_test(%n : index, %A : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %val = memref.load %A[%i] : memref<?xf32>
    %doubled = arith.mulf %val, %c2 : f32
    memref.store %doubled, %A[%i] : memref<?xf32>
  }
  return
}
```
and part of the unrolling:
```
module {
  func.func @unroll_test(%arg0: index, %arg1: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.000000e+00 : f32
    scf.for %arg2 = %c0 to %arg0 step %c1 {
      %0 = arith.subi %arg0, %arg2 : index
      %c4 = arith.constant 4 : index
      %1 = arith.muli %c1, %c4 : index
      %2 = arith.cmpi sge, %0, %1 : index
      scf.if %2 {
        //unrolled
        %3 = memref.load %arg1[%arg2] : memref<?xf32> // use IV
        %4 = arith.mulf %3, %cst : f32
        memref.store %4, %arg1[%arg2] : memref<?xf32>
        %5 = arith.addi %arg2, %c1 : index // update IV

        //unrolled
        %6 = memref.load %arg1[%5] : memref<?xf32> // use updated IV
        %7 = arith.mulf %6, %cst : f32
        memref.store %7, %arg1[%5] : memref<?xf32>
        %8 = arith.addi %5, %c1 : index // update IV
...
```
You'll notice that each of the calls to `arith.mulf` and `memref.store` are updated with new arguments in each unroll instance. This for loop doesn't actually have any iter args (if it did, they'd be specified with something like `iter_args(%acc = %init)`), but when we update the induction variable (each `arith.addi` call), the `mapping` IRMapping ensures that we clone the `memref.load` op with the updated IV as its argument (the IV is used to index into the array, which in this case is `%arg1`). 


The rest of the code in the for loop above:
```
auto yieldOp = mlir::cast<mlir::scf::YieldOp>(
    forOp.getBody()->getTerminator());
mlir::SmallVector<mlir::Value, 6> yieldOperands =
    llvm::to_vector(llvm::map_range(
        yieldOp.getResults(), [&](mlir::Value v) {
          return mapping.lookupOrDefault(v);
        }));

if (i == unrollFactor - 1) {
  finalYieldOperands = yieldOperands;
} else {
  currentIterArgs = yieldOperands;
  currentIV = thenBuilder.create<mlir::arith::AddIOp>(
      loc, currentIV, forOp.getStep());
}
```
updates the induction variable (`currentIV` gets incremented by `forOp.getStep()` in the final `else` block; it's initially known from `forOp.getLowerBound()`) and the loop-carried dependencies (`currentIterArgs` is set to `yieldOperands`). 

The yield op (implicitly inserted by MLIR) passes values to the next iteration of the loop (these are things we update on each loop iteration, the loop-carried dependencies). We can retrieve its operands via `yieldOp.getResults()`, which in this case are `%val` and `%doubled` in our original for loop. So for each of these, we look up the latest value in `mapping` and insert into our SmallVector of values `yieldOperands`. (the first value for our iter args comes from the `forOp.getInitArgs()`)

Then we finally 
```
thenBuilder.create<mlir::scf::YieldOp>(loc, finalYieldOperands);
```

### Running the pass
Bazel build the `optimizer` binary, then run:
```
bazel-bin/optimizer tests/loop_unroll_example.mlir -p loop-unrolling -o loop_unrolled.mlir
```

### TODOs and more on loop unrolling
This isn't a full-on unrolling pass just yet. I've chosen a fixed unroll factor (4) to get the basics working. I still want to add something to determinte trip count and apply full unroll, and perhaps some logic to determine whether and how much to unroll depending on standard heuristics (described below). 

Usually compilers will use a few heuristics (given the tradeoffs inherent to loop unrolling) to determine the appropriate unroll factor. 

You do a few things:
- Estimate loop's tripcount. You can calculate this directly when your bounds are constant. When they're variable, compilers can still generate code that performs check at runtime to see how many times the loop will run (the code in `loop_unroll_pass.cpp` for calculating `remainingIters`, `unrollThreshold`, and `unrollCondition` do this in my example). 
- Look at the loop body to determine number of instructions, memory footprint, data dependencies. 
- Consider the target architecture: look at instruction cache size, # registers available, and SIMD capabilities (for parallelizing calculations). 

With all of this information, you can apply heuristics: for loops with small tripcounts or small bodies, you could apply a large unroll factor or do a full unroll; loops w/ large bodies might impose instruction cache pressure. If you unroll too much, you might end up with [_register spilling_](https://www.cs.cmu.edu/afs/cs.cmu.edu/user/tcm/www/thesis/subsubsection2_10_2_3_1.html). 

If you have a vector register, you can choose an unroll factor that aligns w/ the vector register size, or one that aligns memory accesses to cache line boundaries. 

## Loop Fusion
### Overview
The goal of loop fusion is exactly what it sounds like: given code with multiple loops, like 
```
func.func @loop_fusion_example(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  scf.for %i = %c0 to %c10 step %c1 {
    %0 = memref.load %arg0[%i] : memref<?xf32>
    %1 = arith.addf %0, %0 : f32
    memref.store %1, %arg1[%i] : memref<?xf32>
  }

  scf.for %i = %c0 to %c10 step %c1 {
    %2 = memref.load %arg1[%i] : memref<?xf32>
    %3 = arith.mulf %2, %2 : f32
    memref.store %3, %arg2[%i] : memref<?xf32>
  }

  return
}
```
our goal is to fuse the two loops into one if it's valid to do so. The fused loop for the above example looks like this:
```
scf.for %arg3 = %c0 to %c10 step %c1 {
  %0 = memref.load %arg0[%arg3] : memref<?xf32>
  %1 = arith.addf %0, %0 : f32
  memref.store %1, %arg1[%arg3] : memref<?xf32>
  %2 = memref.load %arg1[%arg3] : memref<?xf32>
  %3 = arith.mulf %2, %2 : f32
  memref.store %3, %arg2[%arg3] : memref<?xf32>
}
```
Intuitively, this is a valid fusion: the memory accesses in our loops are to different function arguments and the computations in each loop are done on those different memory accesses, so we don't have any dependencies between the two loops that might make a fusion semantically incorrect. 


Generally, we get a few benefits out of loop fusion:

__Reduced loop overhead__: We have to do less loop variable initialization, loop condition checking, loop counter incrementing/decrementing, jumps to the start of the loop

__Improved cache locality__: also fairly familiar, but to use an example, 
```
for (int i = 0; i < 1000; i++) {
    a[i] = b[i] + 1;
}

for (int i = 0; i < 1000; i++) {
    c[i] = a[i] * 2;
}
```
`a[i]` might be evicted from the cache between the write in the first loop and the read in the second loop (especially if `a` is large). 

__Decreased code size__: self-explanatory. From this we'll get better instruction cache utilization, reduced memory usage, faster program load times. 

__Other optimizations__: we might be able to more easily optimize code in loops once fused, e.g. CSE opportunities may show up, more vectorization opportunities, better constant propagation

All that said, classic issues loop fusion can introduce are increased register pressure (risk of spilling), worse cache performance (if working set > cache size), potentially harder to parallelize. 

### Theory / Explanation
Actually fusing loops involves a few steps:
1. Identify candidates: if adjacent loops have the same iteration steps (bounds and step, e.g. iterating from 0 to 10 w/ step 1), we might be able to fuse. 
2. Dependency analysis: Fusion needs to preserve data dependencies. There are three types (source [here](https://en.wikipedia.org/wiki/Data_dependency#:~:text=in%20this%20example.-,Anti%2Ddependency%20(write%2Dafter%2Dread),%2Dread%20(WAR)%20hazard)): flow dependence (first loop writes, second reads), anti-dependence (first loop reads, second writes), and output dependence (first loop writes, second writes). 
3. Fuse: exactly what it sounds like. In MLIR (see implementation) we create a new `scf.for` op with the same bounds and step size, clone bodies of both loops into the new loop, update IV uses in the second loop body (since they'll be referring to their old IVs, which aren't the same as those in the newly-created loop body), then remove original loops. Will also need to update operand references. 

### Running the pass
Same as above, then:
```
bazel-bin/optimizer tests/loop_fusion_example.mlir -p loop-fusion -o loops_fused.mlir
```

## Loop Interchange
I wrote this pass using polyhedral analysis for loops, and I'll explain that in `polyhedral_analysis.md`. The tl;dr of this pass is that we might get better cache locality (depending on access patterns) by changing the order of nested loops in e.g. a matmul. 

## Other topics


