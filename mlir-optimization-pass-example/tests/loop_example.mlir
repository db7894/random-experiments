func.func @simple_loop(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %arg0 step %c1 {
    scf.for %j = %c0 to %arg1 step %c1 {
      %sum = arith.addi %i, %j : index
      %mul = arith.muli %sum, %arg2 : index
      // Placeholder for some computation
    }
  }
  return
}