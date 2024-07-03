// https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/7/

#ifndef OPTIMIZATION_PASS_H
#define OPTIMIZATION_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLoopOptimizationPass();

#endif // OPTIMIZATION_PASS_H