# Running passes in this folder
## Purpose
Writing a few MLIR passes when I have time, for learning's sake. 

## Setup notes
Jeremy Kun has a very helpful guide on setting up LLVM that I used to create the initial `WORKSPACE` and `bazel/` setup: https://www.jeremykun.com/2023/08/10/mlir-getting-started/

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
bazel-bin/optimizer tests/licm_example.mlir -o licm_out.mlir
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