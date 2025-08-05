// RUN: asl-opt %s | asl-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = asl.foo %{{.*}} : i32
        %res = asl.foo %0 : i32
        return
    }
}
