#include "detect_loop_pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

struct DetectLoopPass
    : public mlir::PassWrapper<DetectLoopPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();

    f.walk([&](mlir::Operation *op) {
      if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
        llvm::errs() << "Found a loop in function " << f.getName() << ": "
                     << forOp << "\n";
        // Here you would implement your loop optimization logic
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createDetectLoopPass() {
  return std::make_unique<DetectLoopPass>();
}

// struct DetectLoopPass
//     : public mlir::PassWrapper<DetectLoopPass,
//     mlir::OperationPass<mlir::ModuleOp>>
// {
//     void runOnOperation() override
//     {
//         mlir::ModuleOp module = getOperation();
//         module.walk([&](mlir::Operation *op)
//                     {
//                         if (auto forOp =
//                         llvm::dyn_cast<mlir::scf::ForOp>(op))
//                         {
//                             llvm::errs() << "Found a loop: " << forOp <<
//                             "\n";
//                         }
//                     });
//     }
// };

// std::unique_ptr<mlir::Pass> createDetectLoopPass()
// {
//     return std::make_unique<DetectLoopPass>();
// }
// struct DetectLoopPass
//     : public mlir::PassWrapper<DetectLoopPass,
//     mlir::OperationPass<mlir::func::FuncOp>>
// {
//     void runOnOperation() override
//     {
//         mlir::func::FuncOp f = getOperation();
//         f.walk([&](mlir::scf::ForOp forOp)
//                {
//                    // Implement your loop optimization logic here
//                    llvm::errs() << "Found a loop: " << forOp << "\n";
//                });
//     }
// };

// std::unique_ptr<mlir::Pass> createDetectLoopPass()
// {
//     return std::make_unique<DetectLoopPass>();
// }