#ifndef INTERCHANGE_PASS_H
#define INTERCHANGE_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLoopInterchangePass();

#endif // INTERCHANGE_PASS_H