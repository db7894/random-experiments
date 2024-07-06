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