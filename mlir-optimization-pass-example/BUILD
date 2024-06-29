filegroup(
    name = "test_mlir",
    srcs = ["tests/loop_example.mlir"],
)

cc_library(
    name = "detect_loop_pass",
    srcs = ["detect_loop_pass.cpp"],
    hdrs = ["detect_loop_pass.h"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "optimizer",
    srcs = ["main.cpp"],
    deps = [
        ":detect_loop_pass",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//llvm:Support",
    ],
    data = [":test_mlir"],
)
