#ifndef LOOP_FUSION_PASS_H
#define LOOP_FUSION_PASS_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createLoopFusionPass();

#endif // LOOP_FUSION_PASS_H
