# Running passes in this folder
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
