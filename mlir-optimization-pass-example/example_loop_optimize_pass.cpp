#include "example_loop_optimize_pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

/*
Some sources
- https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/loop-reduction/
- https://www.cs.cmu.edu/afs/cs/academic/class/15745-s06/web/handouts/06.pdf
-
https://www.cs.cmu.edu/afs/cs/academic/class/15745-s19/www/lectures/L9-LICM.pdf

*/

// TODO: move this to a header / shared file
static llvm::cl::opt<bool>
    debugPass("debug-pass-opt",
              llvm::cl::desc("Enable debug output, e.g. dumps."),
              llvm::cl::init(false));

#define DEBUG(X)                                                               \
  if (debugPass) {                                                             \
    X;                                                                         \
  }

struct LoopOptimizationPass
    : public mlir::PassWrapper<LoopOptimizationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  /*

  Simple SSA dominance based on reverse postorder traversal and simple
  heuristics (all ops before a given op in postorder dominate it).

  Doesn't handle complicated predicates yet.

  see also:
    - https://mlir.llvm.org/doxygen/Dominance_8cpp_source.html
    - https://llvm.org/doxygen/GenericDomTreeConstruction_8h_source.html

  Algorithm info:
    - https://www.cs.utexas.edu/~pingali/CS380C/2023/lectures/ssa/Dominators.pdf
    - also slide 8:
  https://piazza.com/class_profile/get_resource/ixkj4uoml3y6qd/izeb0ekl6gsn8

  */
  llvm::DenseMap<mlir::Operation *, llvm::SmallPtrSet<mlir::Operation *, 8>>
  computeDominators(mlir::Operation *rootOp) {
    llvm::DenseMap<mlir::Operation *, llvm::SmallPtrSet<mlir::Operation *, 8>>
        dominators;
    llvm::SmallVector<mlir::Operation *, 16> postorder;

    // post-order taversal
    std::function<void(mlir::Operation *)> dfs = [&](mlir::Operation *op) {
      for (mlir::Region &region : op->getRegions()) {
        for (mlir::Block &block : region) {
          for (mlir::Operation &childOp : block) {
            dfs(&childOp);
          }
        }
      }
      postorder.push_back(op);
    };
    dfs(rootOp);

    // init dominator sets
    llvm::SmallPtrSet<mlir::Operation *, 8> allNodes;
    for (mlir::Operation *op : postorder) {
      allNodes.insert(op);
    }
    for (mlir::Operation *op : postorder) {
      dominators[op] = allNodes;
    }
    dominators[rootOp].clear();
    dominators[rootOp].insert(rootOp);

    auto getPredecessors = [](mlir::Operation *op) {
      llvm::SmallVector<mlir::Operation *, 4> predecessors;

      if (op == &op->getBlock()->front()) {
        // If first op in block, predcessor is parent op
        if (mlir::Operation *parentOp = op->getBlock()->getParentOp()) {
          predecessors.push_back(parentOp);
        }
      } else {
        // otherwise predecessor is prev op IN the block
        predecessors.push_back(op->getPrevNode());
      }

      // handle ctrl flow for SCF ops
      if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        // predcessor of for loop body is for operation itself
        predecessors.push_back(forOp.getOperation());
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        // predcessor of then/else blocks is if op
        predecessors.push_back(ifOp.getOperation());
      }
      // TODO more ctrl flow support

      return predecessors;
    };

    // Compute dominators — do/while part of algorithm
    bool changed;
    do {
      changed = false;
      for (auto it = std::next(postorder.rbegin()); it != postorder.rend();
           ++it) {
        mlir::Operation *op = *it;
        llvm::SmallPtrSet<mlir::Operation *, 8> newDom = allNodes;

        auto predecessors = getPredecessors(op);

        // Intersect dominators of all predecessors
        for (mlir::Operation *pred : predecessors) {
          llvm::SmallPtrSet<mlir::Operation *, 8> intersection;
          for (mlir::Operation *domOp : newDom) {
            if (dominators[pred].count(domOp)) {
              intersection.insert(domOp);
            }
          }
          newDom = intersection;
        }

        newDom.insert(op);

        if (newDom != dominators[op]) {
          dominators[op] = newDom;
          changed = true;
        }
      }
    } while (changed);

    DEBUG(llvm::errs() << "Computed dominator sets:\n");
    for (const auto &entry : dominators) {
      DEBUG(llvm::errs() << "Operation: "; entry.first->dump());
      DEBUG(llvm::errs() << "  Dominators:\n");
      for (auto *domOp : entry.second) {
        DEBUG(llvm::errs() << "\t"; domOp->dump());
      }
    }

    return dominators;
  }

  void runOnOperation() override {
    mlir::func::FuncOp f = getOperation();
    auto dominators = computeDominators(f);

    f.walk([&](mlir::scf::ForOp forOp) {
      DEBUG(llvm::errs() << "Analyzing loop: " << forOp << "\n";);

      auto dominatorsForOp = computeDominators(forOp);
      DEBUG(llvm::errs() << "Printing dominstors for forOp...\n";
            for (auto dominator
                 : dominatorsForOp.at(forOp)) {
              llvm::errs() << "Dumping dominator..\n";
              dominator->dump();
            });

      llvm::SmallVector<mlir::Operation *, 4> invariantOps;
      analyzeLoopBody(forOp, dominators, invariantOps);

      DEBUG(llvm::errs() << "Invariant operations found: "
                         << invariantOps.size() << "\n");
      for (auto *op : invariantOps) {
        DEBUG(llvm::errs() << "  "; op->dump());
      }

      for (auto *op : llvm::reverse(invariantOps)) {
        bool canMove = true;
        for (auto &use : op->getUses()) {
          if (forOp->isAncestor(use.getOwner()) &&
              !llvm::is_contained(invariantOps, use.getOwner())) {
            canMove = false;
            break;
          }
        }
        if (canMove) {
          op->moveBefore(forOp);
        } else {
          llvm::errs() << "  Cannot move operation due to dependent uses: "
                       << *op << "\n";
        }
      }
    });
  }

private:
  bool checkOperationInsideLoop(mlir::Operation *op) {
    while (op) {
      if (isa<mlir::scf::ForOp>(op)) {
        return true;
      }
      op = op->getParentOp();
    }
    return false;
  }

  void analyzeLoopBody(
      mlir::scf::ForOp forOp,
      const llvm::DenseMap<mlir::Operation *,
                           llvm::SmallPtrSet<mlir::Operation *, 8>> &dominators,
      llvm::SmallVectorImpl<mlir::Operation *> &invariantOps) {
    // Get the loop body block
    mlir::Block &body = forOp.getRegion().front();

    DEBUG(llvm::errs() << "Analyzing loop body:\n"; forOp.dump());

    // Iterate over all operations in the body except the terminator
    // (scf.yield)
    for (auto &op : llvm::make_range(body.begin(), std::prev(body.end()))) {
      if (auto nestedForOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        // Recursively analyze nested loops
        analyzeLoopBody(nestedForOp, dominators, invariantOps);
        continue;
      }

      DEBUG(llvm::errs() << "Checking operation: "; op.dump());

      bool isLoopInvariant = true;
      for (mlir::Value operand : op.getOperands()) {
        mlir::Operation *definingOp = operand.getDefiningOp();
        if (definingOp) {
          DEBUG(llvm::errs() << "  Checking operand: "; operand.dump());
          DEBUG(llvm::errs() << "  Defined by: "; definingOp->dump());

          // Find the outermost loop containing the current operation
          mlir::Operation *outermostLoop = forOp.getOperation();
          mlir::Operation *parent = op.getParentOp();
          while (parent && !mlir::isa<mlir::func::FuncOp>(parent)) {
            if (mlir::isa<mlir::scf::ForOp>(parent)) {
              outermostLoop = parent;
            }
            parent = parent->getParentOp();
          }

          // Check if the defining op dominates the outermost loop
          auto outermostLoopDominators = dominators.find(outermostLoop);

          auto doms = outermostLoopDominators->second;
          for (auto dom : doms) {
            DEBUG(llvm::errs() << "Found outermost Loop Dominator ";
                  dom->dump());
          }

          if (outermostLoopDominators == dominators.end() ||
              !outermostLoopDominators->second.count(definingOp)) {
            // The defining op does not dominate the loop
            isLoopInvariant = false;
            DEBUG(llvm::errs() << "  Non-invariant operand: "; operand.dump());
            break;
          } else if (operand == forOp.getInductionVar()) {
            // The induction variable is not loop-invariant
            isLoopInvariant = false;
            DEBUG(llvm::errs() << "  Induction variable is not invariant\n");
            break;
          }
        }
      }

      if (isLoopInvariant) {
        llvm::errs() << "  Loop-invariant operation found: " << op << "\n";
        invariantOps.push_back(&op);
      } else {
        llvm::errs() << "  Loop-variant operation found: " << op << "\n";
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createLoopOptimizationPass() {
  return std::make_unique<LoopOptimizationPass>();
}