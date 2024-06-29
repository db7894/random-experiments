#ifndef OPTIMIZATION_PASS_H
#define OPTIMIZATION_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLoopOptimizationPass();
// Placeholder for future pass declarations

#endif // OPTIMIZATION_PASS_H