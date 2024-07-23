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
    std::cout << "Starting runOnOperation\n";
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

        std::cout << "Found a for loop\n";
        auto nextOp = firstLoop->getNextNode();
        if (!nextOp) {
          std::cout << "No next operation, ending walk\n";
          return mlir::WalkResult::interrupt();
        }

        if (auto secondLoop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                firstLoop->getNextNode())) {
          if (processedLoops.count(secondLoop.getOperation()) > 0) {
            return mlir::WalkResult::advance();
          }

          std::cout << "Found a second for loop\n";
          if (tryFuseLoops(firstLoop, secondLoop)) {
            std::cout << "Successfully fused loops\n";
            changed = true;
            processedLoops.insert(firstLoop.getOperation());
            processedLoops.insert(secondLoop.getOperation());
            return mlir::WalkResult::interrupt(); // Interrupt to restart the
                                                  // walk
          } else {
            std::cout << "Failed to fuse loops\n";
          }
        } else {
          std::cout << "Next operation is not a loop, continuing\n";
        }
        processedLoops.insert(firstLoop.getOperation());
        return mlir::WalkResult::advance();
      });
    }

    std::cout << "Finished runOnOperation\n";
  }

  bool conflictingAccesses(mlir::Operation *firstOp,
                           mlir::Operation *secondOp) {
    std::cout << "Checking for conflicting accesses\n";
    // check for flow dep (RAW)
    if (auto firstStore = mlir::dyn_cast<mlir::memref::StoreOp>(firstOp)) {
      if (auto secondLoad = mlir::dyn_cast<mlir::memref::LoadOp>(secondOp)) {
        // if store/load are to same mlir::memref, we need to check
        // if they might access the same element
        if (firstStore.getMemRef() == secondLoad.getMemRef()) {
          // if indices are identical constants, it's a RAW dependence
          if (indicesAreIdenticalConstants(firstStore, secondLoad)) {
            std::cout << "Found identical constant indices\n";
            return true;
          }

          // check for potential overlap
          if (indicesMightOverlap(firstStore, secondLoad)) {
            std::cout << "Indices might overlap\n";
            return true;
          }
        }
      }
    }
    std::cout << "No conflicting accesses found\n";
    return false;
  }

  // helper fn: check if all indices are identical constants
  bool indicesAreIdenticalConstants(mlir::memref::StoreOp store,
                                    mlir::memref::LoadOp load) {
    auto storeIndices = store.getIndices();
    auto loadIndices = load.getIndices();

    if (storeIndices.size() != loadIndices.size()) {
      return false;
    }

    for (size_t i = 0; i < storeIndices.size(); ++i) {
      auto storeIndex =
          storeIndices[i].getDefiningOp<mlir::arith::ConstantOp>();
      auto loadIndex = loadIndices[i].getDefiningOp<mlir::arith::ConstantOp>();

      if (!storeIndex || !loadIndex || storeIndex != loadIndex) {
        return false;
      }
    }

    return true;
  }

  bool indicesMightOverlap(mlir::memref::StoreOp store,
                           mlir::memref::LoadOp load) {
    /*
    Very simplified implementation. We'd generally need to consider:
      - symbolic analysis fo index expressions
      - loop invariant expressions
      - affine expressions if using AffineDialect

    I'll just assume stuff overlaps unless I can prove they're different for
    now.
    */
    auto storeIndices = store.getIndices();
    auto loadIndices = load.getIndices();

    if (storeIndices.size() != loadIndices.size()) {
      return true; // conservative
    }

    for (size_t i = 0; i < storeIndices.size(); ++i) {
      auto storeIndex =
          storeIndices[i].getDefiningOp<mlir::arith::ConstantOp>();
      auto loadIndex = loadIndices[i].getDefiningOp<mlir::arith::ConstantOp>();

      if (!storeIndex || !loadIndex || storeIndex != loadIndex) {
        return false;
      }
    }

    return false;
  }

  bool tryFuseLoops(mlir::scf::ForOp firstLoop, mlir::scf::ForOp secondLoop) {
    std::cout << "Attempting to fuse loops\n";
    // Check if loops have the same bounds and step
    if (firstLoop.getLowerBound() != secondLoop.getLowerBound() ||
        firstLoop.getUpperBound() != secondLoop.getUpperBound() ||
        firstLoop.getStep() != secondLoop.getStep()) {
      std::cout << "Loops have different bounds or step\n";
      return false;
    }

    // Collect memory accesses
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

    std::cout << "Checking for dependencies between loops\n";
    for (auto *firstOp : firstLoopAccesses) {
      for (auto *secondOp : secondLoopAccesses) {
        if (conflictingAccesses(firstOp, secondOp)) {
          std::cout << "Found conflicting accesses, cannot fuse\n";
          return false;
        }
      }
    }

    std::cout << "Merging loop bodies\n";
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

    std::cout << "Updating SSA form\n";
    fusedLoop.walk([&](mlir::Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        if (mlir::Value mappedValue = mapping.lookupOrNull(operand.get()))
          operand.set(mappedValue);
      }
    });

    std::cout << "Replacing uses of original loops' results\n";
    for (auto [oldResult, newResult] :
         llvm::zip(firstLoop.getResults(), fusedLoop.getResults()))
      oldResult.replaceAllUsesWith(newResult);
    for (auto [oldResult, newResult] :
         llvm::zip(secondLoop.getResults(), fusedLoop.getResults()))
      oldResult.replaceAllUsesWith(newResult);

    std::cout << "Removing original loops\n";
    firstLoop.erase();
    secondLoop.erase();

    std::cout << "Successfully fused loops\n";
    return true;
  }
};

std::unique_ptr<mlir::Pass> createLoopFusionPass() {
  return std::make_unique<LoopFusionPass>();
}
