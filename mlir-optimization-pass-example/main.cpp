#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "detect_loop_pass.h"
#include "example_loop_optimize_pass.h"
#include "loop_fusion_pass.h"
#include "loop_interchange_polyhedral.h"
#include "loop_interchange_scheduled.h"
#include "loop_unroll_pass.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> passOption("p", llvm::cl::desc("pass to run"),
                                             llvm::cl::init("-"));

/*
To run:

bazel build //:optimizer
bazel-bin/optimizer tests/loop_example.mlir -o output.mlir
*/

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::registerAllPasses();

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Optimization Pass\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Parse the input file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error parsing input file\n";
    return 1;
  }

  // Apply the optimization pass.
  mlir::PassManager pm(&context);
  // pm.addNestedPass<mlir::func::FuncOp>(createDetectLoopPass());

  if (passOption == "loop-optimization")
    pm.addNestedPass<mlir::func::FuncOp>(createLoopOptimizationPass());
  else if (passOption == "loop-unrolling")
    pm.addNestedPass<mlir::func::FuncOp>(createLoopUnrollingPass());
  else if (passOption == "loop-fusion")
    pm.addNestedPass<mlir::func::FuncOp>(createLoopFusionPass());
  else if (passOption == "loop-interchange")
    pm.addNestedPass<mlir::func::FuncOp>(createLoopInterchangePass());
  else if (passOption == "scheduled-loop-interchange")
    pm.addNestedPass<mlir::func::FuncOp>(createScheduledLoopInterchangePass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Error applying optimization pass\n";
    return 1;
  }

  // Output the result.
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  module->print(output->os());
  output->keep();

  return 0;
}

// int main(int argc, char **argv)
// {
//     mlir::MLIRContext context;
//     mlir::OwningOpRef<mlir::ModuleOp> module;

//     llvm::InitLLVM y(argc, argv);
//     llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Optimization
//     Pass\n");

//     // context.getOrLoadDialect<mlir::StandardOpsDialect>();

//     llvm::errs() << "MLIR Optimization Pass Runner\n";
//     return 0;
// }
