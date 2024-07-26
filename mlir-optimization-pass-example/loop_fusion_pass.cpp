#include "loop_fusion_pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
#include <unordered_set>

// TODO: wrap prints in debug

struct LoopFusionPass
    : public mlir::PassWrapper<LoopFusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    std::cout << "Starting runOnOperation" << std::endl;
    mlir::func::FuncOp funcOp = getOperation();

    bool changed = true;
    std::unordered_set<mlir::Operation *> processedLoops;

    while (changed) {
      changed = false;
      processedLoops.clear();

      funcOp.walk([&](mlir::scf::ForOp firstLoop) {
        if (processedLoops.count(firstLoop.getOperation()) > 0) {
          return mlir::WalkResult::advance();
        }

        std::cout << "Found a for loop" << std::endl;
        auto nextOp = firstLoop->getNextNode();
        if (!nextOp) {
          std::cout << "No next operation, ending walk" << std::endl;
          return mlir::WalkResult::interrupt();
        }

        if (auto secondLoop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                firstLoop->getNextNode())) {
          if (processedLoops.count(secondLoop.getOperation()) > 0) {
            return mlir::WalkResult::advance();
          }

          std::cout << "Found a second for loop" << std::endl;
          if (tryFuseLoops(firstLoop, secondLoop)) {
            std::cout << "Successfully fused loops" << std::endl;
            changed = true;
            processedLoops.insert(firstLoop.getOperation());
            processedLoops.insert(secondLoop.getOperation());
            return mlir::WalkResult::interrupt(); // Interrupt to restart the
                                                  // walk
          } else {
            std::cout << "Failed to fuse loops" << std::endl;
          }
        } else {
          std::cout << "Next operation is not a loop, continuing" << std::endl;
        }
        processedLoops.insert(firstLoop.getOperation());
        return mlir::WalkResult::advance();
      });
    }

    std::cout << "Finished runOnOperation" << std::endl;
  }

  bool conflictingAccesses(mlir::Operation *firstOp,
                           mlir::Operation *secondOp) {
    std::cout << "Checking for conflicting accesses" << std::endl;
    // check for flow dep (RAW)
    if (auto firstStore = mlir::dyn_cast<mlir::memref::StoreOp>(firstOp)) {
      if (auto secondLoad = mlir::dyn_cast<mlir::memref::LoadOp>(secondOp)) {
        // if store/load are to same mlir::memref, we need to check
        // if they might access the same element
        if (firstStore.getMemRef() == secondLoad.getMemRef() &&
            indicesMightOverlap(firstStore, secondLoad)) {
          std::cout << "found flow dependency" << std::endl;
          return true;
        }
      }
    }

    // check for anti-dependence (WAR)
    if (auto firstLoad = mlir::dyn_cast<mlir::memref::LoadOp>(firstOp)) {
      if (auto secondStore = mlir::dyn_cast<mlir::memref::StoreOp>(secondOp)) {
        if (firstLoad.getMemRef() == secondStore.getMemRef() &&
            indicesMightOverlap(firstLoad, secondStore)) {
          std::cout << "found anti dependency" << std::endl;
          return true;
        }
      }
    }

    // check for output dependence (WAW)
    if (auto firstStore = mlir::dyn_cast<mlir::memref::StoreOp>(firstOp)) {
      if (auto secondStore = mlir::dyn_cast<mlir::memref::StoreOp>(secondOp)) {
        if (firstStore.getMemRef() == secondStore.getMemRef() &&
            indicesMightOverlap(firstStore, secondStore)) {
          std::cout << "found output dependency" << std::endl;
          return true;
        }
      }
    }

    std::cout << "No conflicting accesses found" << std::endl;
    return false;
  }

  bool indicesMightOverlap(mlir::Operation *op1, mlir::Operation *op2) {
    /*
    Very simplified implementation. We'd generally need to consider:
      - symbolic analysis fo index expressions
      - loop invariant expressions
      - affine expressions if using AffineDialect

    The only time I'm explicitly returning false here is if the op indices
    are both loop IVs
    */
    auto getIndices =
        [](mlir::Operation *op) -> llvm::SmallVector<mlir::Value> {
      if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return llvm::SmallVector<mlir::Value>(loadOp.getIndices().begin(),
                                              loadOp.getIndices().end());
      } else if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return llvm::SmallVector<mlir::Value>(storeOp.getIndices().begin(),
                                              storeOp.getIndices().end());
      }
      llvm_unreachable("Unexpected operation type");
    };

    auto indices1 = getIndices(op1);
    auto indices2 = getIndices(op2);

    if (indices1.size() != indices2.size()) {
      std::cout << "dims have different size" << std::endl;
      return true; // conservative assumption: overlap if dimensions differ
    }

    for (size_t i = 0; i < indices1.size(); ++i) {
      if (indices1[i] != indices2[i]) {
        // if the indices are different values, check if they're both loop
        // IVs
        auto isLoopInductionVar = [](mlir::Value v) {
          return v.isa<mlir::BlockArgument>() &&
                 mlir::isa<mlir::scf::ForOp>(v.getParentBlock()->getParentOp());
        };

        if (isLoopInductionVar(indices1[i]) &&
            isLoopInductionVar(indices2[i])) {
          // if both are loop IVs, they don't overlap across
          // loop iterations
          return false;
        }

        // otherwise we assume indices might overlap
        return true;
      }
    }

    return true; // all indices are identical, so definite overlap
  }

  bool tryFuseLoops(mlir::scf::ForOp firstLoop, mlir::scf::ForOp secondLoop) {
    std::cout << "Attempting to fuse loops" << std::endl;
    // check if loops have the same bounds and step
    if (firstLoop.getLowerBound() != secondLoop.getLowerBound() ||
        firstLoop.getUpperBound() != secondLoop.getUpperBound() ||
        firstLoop.getStep() != secondLoop.getStep()) {
      std::cout << "Loops have different bounds or step" << std::endl;
      return false;
    }

    mlir::SmallVector<mlir::Operation *, 8> firstLoopAccesses,
        secondLoopAccesses;
    firstLoop.getBody()->walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::memref::LoadOp>(op) ||
          mlir::isa<mlir::memref::StoreOp>(op))
        firstLoopAccesses.push_back(op);
    });
    secondLoop.getBody()->walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::memref::LoadOp>(op) ||
          mlir::isa<mlir::memref::StoreOp>(op))
        secondLoopAccesses.push_back(op);
    });

    std::cout << "Checking for dependencies between loops" << std::endl;
    for (auto *firstOp : firstLoopAccesses) {
      for (auto *secondOp : secondLoopAccesses) {
        if (conflictingAccesses(firstOp, secondOp)) {
          std::cout << "Found conflicting accesses, cannot fuse" << std::endl;
          return false;
        }
      }
    }

    std::cout << "Merging loop bodies" << std::endl;
    mlir::OpBuilder builder(firstLoop);
    auto fusedLoop = builder.create<mlir::scf::ForOp>(
        firstLoop.getLoc(), firstLoop.getLowerBound(),
        firstLoop.getUpperBound(), firstLoop.getStep());

    mlir::IRMapping mapping;
    mapping.map(firstLoop.getInductionVar(), fusedLoop.getInductionVar());
    mapping.map(secondLoop.getInductionVar(), fusedLoop.getInductionVar());

    builder.setInsertionPointToStart(fusedLoop.getBody());
    for (mlir::Operation &op : firstLoop.getBody()->without_terminator())
      builder.clone(op, mapping);
    for (mlir::Operation &op : secondLoop.getBody()->without_terminator())
      builder.clone(op, mapping);

    std::cout << "Updating operand references" << std::endl;
    fusedLoop.walk([&](mlir::Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        if (mlir::Value mappedValue = mapping.lookupOrNull(operand.get()))
          operand.set(mappedValue);
      }
    });

    std::cout << "Replacing uses of original loops' results" << std::endl;
    for (auto [oldResult, newResult] :
         llvm::zip(firstLoop.getResults(), fusedLoop.getResults()))
      oldResult.replaceAllUsesWith(newResult);
    for (auto [oldResult, newResult] :
         llvm::zip(secondLoop.getResults(), fusedLoop.getResults()))
      oldResult.replaceAllUsesWith(newResult);

    std::cout << "Removing original loops" << std::endl;
    firstLoop.erase();
    secondLoop.erase();

    std::cout << "Successfully fused loops" << std::endl;
    return true;
  }
};

std::unique_ptr<mlir::Pass> createLoopFusionPass() {
  return std::make_unique<LoopFusionPass>();
}
