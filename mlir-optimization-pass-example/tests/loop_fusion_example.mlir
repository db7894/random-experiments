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