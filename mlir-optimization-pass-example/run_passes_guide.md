# Running passes in this folder
## Loop Detection Pass
This pass basically does nothing. It just checks if you have a loop from the `scf` dialect and spits it back out if so. 
You can run it with the following two commands, using the `loop_example.mlir` in the `tests/` directory or using your own example. 
```
bazel build //:optimizer
bazel-bin/optimizer tests/loop_example.mlir -o output.mlir
```
