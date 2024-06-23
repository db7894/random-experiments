# Differentiating Programs in Haskell
This is a basic port of the ocaml code from [this Jane Street blog](https://blog.janestreet.com/computations-that-differentiate-debug-and-document-themselves/). 

I created a `Computation` module in `src/Coputation.hs` and a test in `test/ComputationTest.hs`. 

## Cabal setup
First initialized cabal with `cabal init --interactive` and modified my `haskell-diff.cabal` file to include necessary deps, e.g. `containers` under `build-depends` so I could use `Data.Map.Strict` in my `Computation` module. 

## Building and running tests
Constructed a simple test, exactly the same as in the Jane Street blog, to verify (a) the computation executed correctly to `729` with given inputs and that partial derivatives came out correct. 

I first ran `cabal build`, then `cabal exec -- ghc -o computation-test test/ComputationTest.hs` (the `--` is similar to the `bazel run` quirk). 

Finally, I ran the test binary:
```
danielbashir@Daniels-MBP haskell-diff % ./computation-test 
Cases: 1  Tried: 1  Errors: 0  Failures: 0
```