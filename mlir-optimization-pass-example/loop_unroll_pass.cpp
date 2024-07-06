/*
sources (no I haven't read all of these yet):
- https://yashwantsingh.in/posts/loop-unroll/
- https://joshpeterson.github.io/learning-loop-unrolling
- https://www.modular.com/blog/what-is-loop-unrolling-how-you-can-speed-up-mojo
- https://mlir.llvm.org/doxygen/LoopUnroll_8cpp_source.html
- https://mlir.llvm.org/doxygen/LoopUtils_8cpp_source.html#l00882
- trip count: https://mlir.llvm.org/doxygen/LoopAnalysis_8cpp_source.html#l00088
- and https://mlir.llvm.org/doxygen/LoopAnalysis_8cpp_source.html#l00038

I'll first try to implement a *Full* unroll pass, then do peeling / partial
unroll.
*/

#include "loop_unroll_pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

struct LoopUnrollingPass
    : public mlir::PassWrapper<LoopUnrollingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    getOperation().walk([&](mlir::scf::ForOp forOp) {
      if (forOp->getParentOfType<mlir::scf::ForOp>())
        return;
      if (shouldUnroll(forOp))
        unrollLoop(forOp);
    });
  }

private:
  const int unrollFactor = 4; // TODO: determine this dynamically.

  bool shouldUnroll(mlir::scf::ForOp forOp) {
    // unroll if we can determine the trip count

    // Check if lower bound, upper bound, and step are constant
    auto lb =
        forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto ub =
        forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto step = forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();

    llvm::errs() << "Lower bound is constant: " << (lb != nullptr) << "\n";
    llvm::errs() << "Upper bound is constant: " << (ub != nullptr) << "\n";
    llvm::errs() << "Step is constant: " << (step != nullptr) << "\n";

    // If upper bound is not constant, we can still unroll if we add a runtime
    // check
    if (!lb || !step)
      return false;

    // Calculate trip count
    if (ub) {
      int64_t tripCount =
          (ub.value() - lb.value() + step.value() - 1) / step.value();
      llvm::errs() << "Trip count: " << tripCount << "\n";
      return tripCount > 1;
    } else {
      // We can unroll with a runtime check
      llvm::errs() << "Can unroll with runtime check\n";
      return true;
    }
  }

  void unrollLoop(mlir::scf::ForOp forOp) {
    mlir::OpBuilder builder(forOp);
    mlir::Location loc = forOp.getLoc();

    auto newForOp = builder.create<mlir::scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        forOp.getInitArgs(),
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv,
            mlir::ValueRange iterArgs) {
          // Calculate remaining iterations
          mlir::Value remainingIters =
              b.create<mlir::arith::SubIOp>(loc, forOp.getUpperBound(), iv);
          mlir::Value unrollThreshold = b.create<mlir::arith::MulIOp>(
              loc, forOp.getStep(),
              b.create<mlir::arith::ConstantIndexOp>(loc, unrollFactor));
          mlir::Value unrollCondition = b.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::sge, remainingIters,
              unrollThreshold);

          b.create<mlir::scf::IfOp>(
              loc, unrollCondition,
              [&](mlir::OpBuilder &thenBuilder, mlir::Location thenLoc) {
                mlir::Value currentIV = iv;
                mlir::ValueRange currentIterArgs = iterArgs;
                mlir::SmallVector<mlir::Value, 6> finalYieldOperands;

                llvm::errs() << "PRINTING CURRENT IV AND ITER ARGS\n";
                llvm::errs() << "IV IS\n";
                iv.dump();
                llvm::errs() << "ITER ARGS IS\n";
                for (auto iterArg : currentIterArgs)
                  iterArg.dump();

                for (int i = 0; i < unrollFactor; ++i) {
                  mlir::IRMapping mapping;
                  mapping.map(forOp.getInductionVar(), currentIV);
                  mapping.map(forOp.getRegionIterArgs(), currentIterArgs);

                  llvm::errs() << "UPDATED MLIR IR MAPPING: ITERATING...\n";
                  auto valueMap = mapping.getValueMap();
                  for (auto it = valueMap.begin(); it != valueMap.end(); ++it) {
                    mlir::Value src = it->first;
                    mlir::Value dest = it->second;

                    // Print the source value
                    llvm::outs() << "Source Value: ";
                    src.print(llvm::outs());
                    llvm::outs() << "\n";

                    // Print the destination value
                    llvm::outs() << "Destination Value: ";
                    dest.print(llvm::outs());
                    llvm::outs() << "\n";
                  }

                  for (mlir::Operation &op :
                       forOp.getBody()->without_terminator()) {
                    thenBuilder.clone(op, mapping);
                  }

                  auto yieldOp = mlir::cast<mlir::scf::YieldOp>(
                      forOp.getBody()->getTerminator());
                  mlir::SmallVector<mlir::Value, 6> yieldOperands =
                      llvm::to_vector(llvm::map_range(
                          yieldOp.getResults(), [&](mlir::Value v) {
                            return mapping.lookupOrDefault(v);
                          }));

                  if (i == unrollFactor - 1) {
                    finalYieldOperands = yieldOperands;
                  } else {
                    currentIterArgs = yieldOperands;
                    currentIV = thenBuilder.create<mlir::arith::AddIOp>(
                        loc, currentIV, forOp.getStep());
                  }
                }

                thenBuilder.create<mlir::scf::YieldOp>(loc, finalYieldOperands);
              },
              [&](mlir::OpBuilder &elseBuilder, mlir::Location elseLoc) {
                // If we can't unroll, just clone original loop body
                mlir::IRMapping mapping;
                mapping.map(forOp.getInductionVar(), iv);
                mapping.map(forOp.getRegionIterArgs(), iterArgs);

                for (mlir::Operation &op :
                     forOp.getBody()->without_terminator()) {
                  elseBuilder.clone(op, mapping);
                }

                auto yieldOp = mlir::cast<mlir::scf::YieldOp>(
                    forOp.getBody()->getTerminator());
                elseBuilder.create<mlir::scf::YieldOp>(
                    yieldOp.getLoc(),
                    llvm::to_vector(llvm::map_range(
                        yieldOp.getResults(), [&](mlir::Value v) {
                          return mapping.lookupOrDefault(v);
                        })));
              });

          auto yieldOp =
              mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
          b.create<mlir::scf::YieldOp>(loc, yieldOp.getResults());
        });

    forOp.replaceAllUsesWith(newForOp);
    forOp.erase();
  }
};

std::unique_ptr<mlir::Pass> createLoopUnrollingPass() {
  return std::make_unique<LoopUnrollingPass>();
}