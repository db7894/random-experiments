#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "isl/isl-noexceptions.h"
#include <map>
#include <set>
#include <vector>

using namespace mlir;

struct AffineExpression {
  std::vector<int> coefficients;
  int constant;

  int evaluate(const std::vector<int> &iterationVector) const {
    int result = constant;
    for (size_t i = 0; i < coefficients.size(); ++i) {
      result += coefficients[i] * iterationVector[i];
    }
    return result;
  }
};

struct AccessFunction {
  std::vector<AffineExpression> dimensions;
};

class IterationDomain {
public:
  void addConstraint(const AffineExpression &lowerBound,
                     const AffineExpression &upperBound) {
    constraints.push_back({lowerBound, upperBound});
  }

  bool contains(const std::vector<int> &point) const {
    for (const auto &constraint : constraints) {
      if (constraint.first.evaluate(point) >
          constraint.second.evaluate(point)) {
        return false;
      }
    }
    return false;
  }

  size_t getNumDimensions() const {
    return constraints.empty() ? 0 : constraints[0].first.coefficients.size();
  }

  const std::vector<std::pair<AffineExpression, AffineExpression>> &
  getConstraints() const {
    return constraints;
  }

private:
  std::vector<std::pair<AffineExpression, AffineExpression>> constraints;
};

class Dependence {
public:
  std::vector<int> source;
  std::vector<int> target;
  int type; // 0: RAW, 1: WAR, 2: WAW
};

class DependenceAnalysis {
public:
  void addAccess(const AccessFunction &access, bool isWrite,
                 const std::string &arrayName) {
    accesses.push_back({access, isWrite, arrayName});
  }

  std::vector<Dependence> construct(const IterationDomain &domain) {
    std::vector<Dependence> dependencies;
    try {
      isl::ctx ctx = isl::ctx(isl_ctx_alloc());

      for (size_t i = 0; i < accesses.size(); ++i) {
        for (size_t j = i + 1; j < accesses.size(); ++j) {
          auto deps = findDependencies(accesses[i], accesses[j], domain, ctx);
          dependencies.insert(dependencies.end(), deps.begin(), deps.end());
        }
      }
      ctx.release();
    } catch (const std::exception &e) {
      llvm::errs() << "Exception in DependenceAnalysis::construct: " << e.what()
                   << "\n";
    }
    return dependencies;
  }

private:
  std::vector<std::tuple<AccessFunction, bool, std::string>> accesses;

  std::vector<Dependence>
  findDependencies(const std::tuple<AccessFunction, bool, std::string> &access1,
                   const std::tuple<AccessFunction, bool, std::string> &access2,
                   const IterationDomain &domain, isl::ctx &ctx) {
    std::vector<Dependence> deps;

    // dependency check
    const auto &[func1, isWrite1, name1] = access1;
    const auto &[func2, isWrite2, name2] = access2;

    // different arrays, no dep
    if (name1 != name2)
      return deps;

    // read-read --> no dep
    if (!isWrite1 && !isWrite2)
      return deps;

    try {
      // create ISL sets, maps
      isl::set iterationSpace = createIterationSpace(domain, ctx);
      isl::map access1Map = createAccessMap(func1, name1, ctx);
      isl::map access2Map = createAccessMap(func2, name2, ctx);

      // compute deps
      isl::map dep;
      if (isWrite1 && !isWrite2) {
        // RAW dependence (read after write)
        dep = access1Map.reverse()
                  .apply_range(access2Map)
                  .intersect_domain(iterationSpace)
                  .intersect_range(iterationSpace);
      } else if (!isWrite1 && isWrite2) {
        // WAR dependence
        dep = access1Map.apply_range(access2Map.reverse())
                  .intersect_domain(iterationSpace)
                  .intersect_range(iterationSpace);
      } else if (isWrite1 && isWrite2) {
        // WAW dependence
        dep = access1Map.reverse()
                  .apply_range(access2Map)
                  .intersect_domain(iterationSpace)
                  .intersect_range(iterationSpace);
      }

      // extract deps, capture isWrite1 and isWrite2 by ref.
      dep.foreach_basic_map([&deps, isWrite1 = isWrite1, isWrite2 = isWrite2](
                                isl::basic_map bmap) -> isl::stat {
        isl::basic_set sample = bmap.domain();
        if (!sample.is_empty()) {
          isl::point p = sample.sample_point();
          if (!p.is_null()) {
            std::vector<int> source, target;
            unsigned int dim = isl_basic_map_dim(bmap.get(), isl_dim_set);
            for (unsigned int i = 0; i < dim; ++i) {
              source.push_back(
                  p.get_coordinate_val(isl::dim::set, (int)i).get_num_si());
              target.push_back(
                  p.get_coordinate_val(isl::dim::set, (int)i).get_num_si());
            }
            int type =
                isWrite1 ? (isWrite2 ? 2 : 0) : 1; // 0: RAW, 1: WAR, 2: WAW
            deps.emplace_back(Dependence{source, target, type});
          }
        }
        return isl::stat::ok();
      });
    } catch (const std::exception &e) {
      llvm::errs() << "ISL exception in findDependencies: " << e.what() << "\n";
    }

    return deps;
  }

  isl::set createIterationSpace(const IterationDomain &domain, isl::ctx &ctx) {
    std::string setStr = "[N] -> {[";
    int numDims = domain.getNumDimensions();
    for (int i = 0; i < numDims; ++i) {
      setStr += "i" + std::to_string(i);
      if (i < numDims - 1)
        setStr += ", ";
    }
    setStr += "] : 0 <= ";
    for (int i = 0; i < numDims; ++i) {
      setStr += "i" + std::to_string(i) + " < N";
      if (i < numDims - 1)
        setStr += " and 0 <= ";
    }
    setStr += "}";
    llvm::errs() << "Iteration Space: " << setStr << "\n";
    return isl::set(ctx, setStr);
  }

  isl::map createAccessMap(const AccessFunction &func,
                           const std::string &arrayName, isl::ctx &ctx) {
    std::string mapStr = "[N] -> {[";
    int numDims = func.dimensions[0].coefficients.size();
    for (int i = 0; i < numDims; ++i) {
      mapStr += "i" + std::to_string(i);
      if (i < numDims - 1)
        mapStr += ", ";
    }
    mapStr += "] -> " + arrayName + "[";
    for (size_t i = 0; i < func.dimensions.size(); ++i) {
      mapStr += formatAffineExpression(func.dimensions[i]);
      if (i < func.dimensions.size() - 1)
        mapStr += ", ";
    }
    mapStr += "] : 0 <= ";
    for (int i = 0; i < numDims; ++i) {
      mapStr += "i" + std::to_string(i) + " < N";
      if (i < numDims - 1)
        mapStr += " and 0 <= ";
    }
    mapStr += "}";
    llvm::errs() << "Access Map: " << mapStr << "\n";
    return isl::map(ctx, mapStr);
  }

  std::string formatAffineExpression(const AffineExpression &expr) {
    std::string result;
    for (size_t i = 0; i < expr.coefficients.size(); ++i) {
      if (expr.coefficients[i] != 0) {
        if (!result.empty() && expr.coefficients[i] > 0)
          result += " + ";
        if (expr.coefficients[i] == -1)
          result += "-";
        else if (expr.coefficients[i] != 1)
          result += std::to_string(expr.coefficients[i]) + "*";
        result += "i" + std::to_string(i);
      }
    }
    if (expr.constant != 0) {
      if (!result.empty() && expr.constant > 0)
        result += " + ";
      result += std::to_string(expr.constant);
    }
    return result.empty() ? "0" : result;
  }
};

struct LoopInterchangePass
    : public PassWrapper<LoopInterchangePass, OperationPass<func::FuncOp>> {
public:
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    llvm::errs() << "Starting loop interchange pass\n";

    f.walk([&](mlir::affine::AffineForOp outerLoop) {
      llvm::errs() << "Found outer loop\n";
      std::vector<mlir::affine::AffineForOp> loopNest;
      loopNest.push_back(outerLoop);

      // find all nested loops
      auto currentLoop = outerLoop;
      while (true) {
        auto innerLoop = findImmediateInnerLoop(currentLoop);
        if (!innerLoop)
          break;
        loopNest.push_back(innerLoop);
        currentLoop = innerLoop;
      }

      llvm::errs() << "Found " << loopNest.size() << " nested loops\n";

      // proceed if at least two loops
      if (loopNest.size() >= 2) {
        llvm::errs() << "Checking if loops can be interchanged\n";
        if (canInterchangeLoops(loopNest)) {
          llvm::errs() << "Interchanging loops\n";
          auto newLoops = interchangeLoops(loopNest);

          // replace old outermost loop with new outermost loop
          outerLoop.replaceAllUsesWith(newLoops.front());
          outerLoop.erase();
        } else {
          llvm::errs() << "Loops cannot be interchanged\n";
        }
      }
    });

    llvm::errs() << "Finished loop interchange pass\n";
  }

private:
  mlir::affine::AffineForOp
  findImmediateInnerLoop(mlir::affine::AffineForOp loop) {
    for (Operation &op : loop.getBody()->getOperations()) {
      if (mlir::affine::AffineForOp innerLoop =
              dyn_cast<mlir::affine::AffineForOp>(op)) {
        return innerLoop;
      }
    }
    return mlir::affine::AffineForOp();
  }

  bool
  canInterchangeLoops(const std::vector<mlir::affine::AffineForOp> &loops) {
    if (loops.size() < 2) {
      llvm::errs() << "Not enough loops to interchange\n";
      return false;
    }

    IterationDomain domain;

    // handle non-constant loop bounds
    auto addConstraintFromMap = [&domain](AffineMap map, int loopIndex,
                                          size_t totalLoops) {
      std::vector<int> lowerCoeff(totalLoops, 0);
      std::vector<int> upperCoeff(totalLoops, 0);
      lowerCoeff[loopIndex] = 1;
      upperCoeff[loopIndex] = 1;

      AffineExpression lowerBound, upperBound;
      lowerBound.coefficients = lowerCoeff;
      upperBound.coefficients = upperCoeff;

      // assume lower bound is 0 if not specified
      if (map.getNumResults() == 0) {
        lowerBound.constant = 0;
        upperBound.constant = 0; // to be replaced by the upper bound
      } else {
        auto expr = map.getResult(0);
        if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
          lowerBound.constant = 0;
          upperBound.constant = constExpr.getValue();
        } else {
          // for non-constant expressions, can't determine exact bounds
          // so use a placeholder (here we use 0 for lower, 1000000 for upper)
          lowerBound.constant = 0;
          upperBound.constant = 1000000;
        }
      }

      domain.addConstraint(lowerBound, upperBound);
    };

    for (size_t i = 0; i < loops.size(); ++i) {
      addConstraintFromMap(
          const_cast<mlir::affine::AffineForOp &>(loops[i]).getLowerBoundMap(),
          i, loops.size());
      addConstraintFromMap(
          const_cast<mlir::affine::AffineForOp &>(loops[i]).getUpperBoundMap(),
          i, loops.size());
    }

    DependenceAnalysis analysis;

    // analyze memory accesses
    size_t arrayCounter = 0;
    const_cast<mlir::affine::AffineForOp &>(loops[0]).walk(
        [&](memref::LoadOp loadOp) {
          analysis.addAccess(createAccessFunction(loadOp, loops.size()), false,
                             "Array" + std::to_string(arrayCounter++));
        });
    const_cast<mlir::affine::AffineForOp &>(loops[0]).walk(
        [&](memref::StoreOp storeOp) {
          analysis.addAccess(createAccessFunction(storeOp, loops.size()), true,
                             "Array" + std::to_string(arrayCounter++));
        });

    try {
      auto dependencies = analysis.construct(domain);

      // check if interchange legal for any pair of adjacent loops
      for (size_t i = 0; i < loops.size() - 1; ++i) {
        for (const auto &dep : dependencies) {
          if ((dep.source[i] < dep.target[i] &&
               dep.source[i + 1] > dep.target[i + 1]) ||
              (dep.source[i] > dep.target[i] &&
               dep.source[i + 1] < dep.target[i + 1])) {
            return false;
          }
        }
      }
      llvm::errs() << "No dependencies preventing interchange found\n";
      return true;
    } catch (const std::exception &e) {
      llvm::errs() << "Exception in canInterchangeLoops: " << e.what() << "\n";
      return false;
    }
  }

  AccessFunction createAccessFunction(Operation *op, size_t numLoops) {
    AccessFunction access;
    SmallVector<Value, 4> indices;
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      indices = loadOp.getIndices();
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      indices = storeOp.getIndices();
    }

    for (auto index : indices) {
      AffineExpression expr;
      expr.coefficients = std::vector<int>(numLoops, 0); // Assuming 3D loops
      if (auto blockArg = index.dyn_cast<BlockArgument>()) {
        if (blockArg.getArgNumber() < numLoops) {
          expr.coefficients[blockArg.getArgNumber()] = 1;
        }
      } else {
        // if we can't determine the index, use a placeholder
        expr.coefficients = std::vector<int>(numLoops, 1);
      }
      access.dimensions.push_back(expr);
    }
    return access;
  }

  AffineExpression convertAffineMap(AffineMap map) {
    AffineExpression expr;
    expr.coefficients = std::vector<int>(map.getNumInputs(), 0);
    expr.constant = 0;

    for (auto result : map.getResults()) {
      result.walk([&](AffineExpr subexpr) {
        if (auto constExpr = subexpr.dyn_cast<AffineConstantExpr>()) {
          expr.constant += constExpr.getValue();
        } else if (auto dimExpr = subexpr.dyn_cast<AffineDimExpr>()) {
          expr.coefficients[dimExpr.getPosition()] += 1;
        }
        return WalkResult::advance();
      });
    }

    return expr;
  }

  std::vector<mlir::affine::AffineForOp>
  interchangeLoops(const std::vector<mlir::affine::AffineForOp> &loops) {
    if (loops.size() < 2) {
      llvm::errs() << "Not enough loops to interchange\n";
      return {};
    }

    std::vector<mlir::affine::AffineForOp> newLoops;
    try {
      OpBuilder builder(loops[0]);

      // create new loops w/ interchanged bounds
      for (size_t i = 0; i < loops.size(); ++i) {
        size_t sourceIndex = (i + 1) % loops.size(); // rotate loop bounds
        auto newLoop = builder.create<mlir::affine::AffineForOp>(
            const_cast<mlir::affine::AffineForOp &>(loops[i]).getLoc(),
            const_cast<mlir::affine::AffineForOp &>(loops[sourceIndex])
                .getLowerBoundOperands(),
            const_cast<mlir::affine::AffineForOp &>(loops[sourceIndex])
                .getLowerBoundMap(),
            const_cast<mlir::affine::AffineForOp &>(loops[sourceIndex])
                .getUpperBoundOperands(),
            const_cast<mlir::affine::AffineForOp &>(loops[sourceIndex])
                .getUpperBoundMap());
        newLoops.push_back(newLoop);
        builder.setInsertionPointToStart(newLoop.getBody());
      }

      llvm::errs() << "Created loops with interchanged bounds\n";

      // move body of the innermost loop to new innermost loop
      auto &innerLoopOps = const_cast<mlir::affine::AffineForOp &>(loops.back())
                               .getBody()
                               ->getOperations();
      auto &newInnerLoopOps =
          const_cast<mlir::affine::AffineForOp &>(newLoops.back())
              .getBody()
              ->getOperations();
      newInnerLoopOps.splice(newInnerLoopOps.begin(), innerLoopOps,
                             innerLoopOps.begin(),
                             std::prev(innerLoopOps.end()));

      llvm::errs() << "Moved loop body\n";

      // update IV uses
      for (size_t i = 0; i < loops.size(); ++i) {
        const_cast<mlir::affine::AffineForOp &>(loops[i])
            .getInductionVar()
            .replaceAllUsesWith(
                const_cast<mlir::affine::AffineForOp &>(newLoops[i])
                    .getInductionVar());
      }

      llvm::errs() << "Updated induction variable uses\n";

      llvm::errs() << "Successfully interchanged loops\n";
    } catch (const std::exception &e) {
      llvm::errs() << "Exception in interchangeLoops: " << e.what() << "\n";
    }

    return newLoops;
  }
};

std::unique_ptr<mlir::Pass> createLoopInterchangePass() {
  return std::make_unique<LoopInterchangePass>();
}
