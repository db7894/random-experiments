#ifndef DETECT_LOOP_PASS_H
#define DETECT_LOOP_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createDetectLoopPass();

#endif // DETECT_LOOP_PASS_H