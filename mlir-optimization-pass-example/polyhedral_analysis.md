# Polyhedral Analysis
## Verification
Programs spend most of their time in loops — this motivates the need for a variety of loop optimizations. Loops can be complicated to optimize, since we have to worry about things like loop-carried dependencies (one iteration of a loop might depend on the previous iteration), so it's of great benefit to have an analysis tool that lets us formally reason about the legality of program transformations. 

To take an example in pseudocode (walking through [this presentation](https://www.youtube.com/watch?v=_TFrPGV_A-s)):
```
for i in [1...4]:
    A[i] = A[i-1]
```
We can reason intuitively that reversing the loop iteration order, yielding:
```
for i in [4...1]:
    A[i] = A[i-1]
```
would _not_ be a valid transformation of the program: the second iteration of the loop accesses `A[1]`, which is written by the first iteration of the loop (which sets `A[1] = A[0]`). 

The polyhedral (or polytope) model gives us a formal way to do reasoning like this. First, if we refer to the loop body as a statement `S: A[i] = A[i-1]` which can be indexed based on the loop iteration, we can say `S[2] = A[2] - A[1]` and write the set of statements in the original example as `{S[i] \ | 1 <= i < 4}` with `{S[i] -> i}`. The (invalid) transformation would be written as `{S[i] \ | 1 <= i < 4}` where `{S[i] -> 5-i}`.

To account for the iteration dependencies we mentioned earlier, we can consider sets of pairs of statements, such as `\{(S[i], S[i+1]) \ | 1 <= i < 3\}` where the pairing indicates that `S[i+1]` depends on `S[i]`. We'd consider this the set of data dependencies, and a new schedule will be legal if the set of violated data dependencies is empty. 

More formally, the set of all violated dependencies in the new schedule is a set of pairs of statements `(a,b)` where
1. Statement `a` sends data to statement `b`
2. `a` comes after `b` in the new schedule

To check our example, we'd need to verify that the set 
`{(S[i], S[i+1]) \ | \ 1 <= i <= 3 && NewSched(i) >= NewSched(i+1)}`, 
or 
`{(S[i], S[i+1]) \ | \ 1 <= i <= 3 && 5 - i >= 5 - (i+1)}` 
is empty. 

This set if empty if the set of linear constraints
```
1 <= i <= 3
5 - i >= 5 - (i+1)
```
has no solution — we can solve this problem with the well-known technique __Integer Linear Programming__ (ILP). 

In this case, we can notice that `i=1` is a valid solution to the system of equations, and thus there exists a violated dependency in our set. 

Geometrically, executions of a statement in a loopnest are represented by _points_ within a convex integer polyhedron; the boundaries of this polyhedron are formed by loop bounds and other constraints. 

## Example / Code
To use `loop_interchange_polyhedral.cpp` as an example, consider the dependency analysis:

The `construct` method creates a vector of dependencies, and the `findDependencies` sets up our inequalities that we'll use ILP to find a solution for. Consider the example
```
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        A[i][j] = A[i-1][j+1] + B[i][j];
```
We create an _iteration space_, the set of all valid iteration points (valid values of `i` and `j`) and two _access maps_, which represent how array elements are accessed (for the read of `A`, this is `{[i,j] -> A[i-1][j+1]}`). We then compute a dependency map based on the type of potential dependency (RAW, WAR, WAW — see other doc for explanations of these). 

To consider RAW:
```
dep = access1Map.reverse()
          .apply_range(access2Map)
          .intersect_domain(iterationSpace)
          .intersect_range(iterationSpace);
```

This creates a map representing all pairs of iteration points `(i_1, j_1)` and `(i_2, j_2)` where:
- `(i_1, j_1)` writes to `A[i_1][j_1]`
- `(i_2, j_2)` reads from `A[i_1][j_1]`
- both `(i_1, j_1)` and `(i_2, j_2)` are within the iteration space
- `(i_2, j_2)` comes after `(i_1, j_1)` in execution order

In our loopnest example above, we'd get something like `{[i_1,j_1] -> [i_2,j_2] : i_2 = i_1+1 and j_2 = j_1-1 and 0 <= i_1,j_1,i_2,j_2 < N}`. When we iterate over basic maps in `dep`, we're looking at objects that represent relations like this. The domain of that basic map is the set of all source iterations `[i_1,j_1]` that have a dependency, and we then sample a point from this set if it's not empty to get a concrete source iteration, e.g. `[0,1]` representing the iteration where `i=0` and `j=1`. 

We then extract coordinates from this sampled point: in our example `source` would be `[0,1]` and `target` is `[1,0]` based on the transformation we defined for `[i_1,j_1] -> [i_2,j_2]`. 

So, we'd then get a dependency like 
```
{
    source: [0, 1],
    target: [1, 0],
    type: 0 (RAW)
}
```

The construction of our dependencies like with
```
dep = access1Map.reverse()
                  .apply_range(access2Map)
                  .intersect_domain(iterationSpace)
                  .intersect_range(iterationSpace);
```
is where the bulk of ILP happens, using the ISL library. the `reverse()` and `apply_range()` calls find pairs of iterations where the first iteration writes to an array element that the second iteration reads. So we get `{[i_1,j_1] -> [i_2,j_2] : i_1 = i_2+1 and j_1 = j_2-1}`. 

Then, `intersect_domain` and `intersect_range` make sure the source iterations `(i_1,j_1)` and target iterations `(i_2,j_2)` are in valid iteration space. `intersect_domain` adds the constraint `0 <= i_1,j_1 < N` while `intersect_range` adds the constraint `0 <= i_2,j_2 < N`. So, finally, we get `{[i_1,j_1] -> [i_2,j_2] : i_2 = i_1+1 and j_2 = j_1-1 and 0 <= i_1,j_1,i_2,j_2 < N}`.

The final map `dep` is the solution to this sytem of inequalities. 

In the legality check code for interchange:
```
for (size_t i = 0; i < loops.size() - 1; ++i) {
    for (const auto &dep : dependencies) {
        if ((dep.source[i] < dep.target[i] &&
            dep.source[i + 1] > dep.target[i + 1]) ||
            (dep.source[i] > dep.target[i] &&
            dep.source[i + 1] < dep.target[i + 1])) {
        return false;
        }
    }
}
```
using our example above, we'll run the outer loop once and go through each dependency in `dependencies` — consider the example instance `([1,2] -> [0,3])` (based on the `(i_1,j_1) -> (i_2,j_2)` map definition). Here, `dep.source = [1,2]` and `dep.target = [0,3]`. 

With `i=0` the condition
```
if (dep.source[i] < dep.target[i] &&
    dep.source[i + 1] > dep.target[i + 1])
```
will check some inequalities. The relevant ones are:
- `dep.source[0] > dep.target[0]` or `1 > 0` -> true: this is checking that the dependency goes backward in the ith loop (src iteration has a larger index than the target iter)
- `dep.source[1] < dep.target[1]` or `2 < 3` -> false: this is checking that the dependency goes forward in the `(i+1)`th loop (src iter has a smaller idx than the target iter). 

and return false. 

Loop interchange is valid if a dependency goes in the same direction in two adjacent loop dimensions. In this case, since the dependency goes in different directions, the interchange is invalid. 



## Scheduling



## Resources
- https://mlir.llvm.org/docs/Rationale/RationaleSimplifiedPolyhedralForm/
- https://mlir.llvm.org/docs/Rationale/
- https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/polyhedral/
- https://en.wikipedia.org/wiki/Polytope_model
- https://www.infosun.fim.uni-passau.de/cl/papers/concur93c.pdf
- https://www.youtube.com/watch?v=_TFrPGV_A-s
