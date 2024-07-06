#ifndef UNROLL_PASS_H
#define UNROLL_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLoopUnrollingPass();

#endif // UNROLL_PASS_H