// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

pure func Ones{N}() => bits(N)
begin
  return NOT Zeros{N};
end;

constant A = 15;


// CHECK: "asl.func"() <{args = ["w", "x", "y", "z"], args_types = [!asl.int<<1 : i32, [#asl.int_constraint<, 15 : i64>], 0 : i32>>, !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>], builtin = false, name = "parameters_of_expressions", parameters = [{identifier = "B", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "C", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "D", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "E", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "F", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "G", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "H", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "I", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "J", type = !asl.int<<3 : i32, [], 0 : i32>>}], primitive = false, return_type = !asl.bits<15 : i64, []>, subprogram_type = 1 : i32}> ({
// CHECK: ^bb0(%arg0: !asl.int<<1 : i32, [#asl.int_constraint<, 15 : i64>], 0 : i32>>, %arg1: !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, %arg2: !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, %arg3: !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>):
// CHECK:   %1 = "asl.expr.var"() <{name = "B"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %2 = "asl.expr.var"() <{name = "C"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %3 = "asl.expr.var"() <{name = "D"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %4 = "asl.expr.var"() <{name = "E"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %5 = "asl.expr.var"() <{name = "F"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %6 = "asl.expr.var"() <{name = "G"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %7 = "asl.expr.var"() <{name = "H"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %8 = "asl.expr.var"() <{name = "I"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %9 = "asl.expr.var"() <{name = "J"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %10 = "asl.expr.literal.int"() <{value = "15"}> : () -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %11 = "asl.expr.atc.int.range"(%arg0, %10, %1) : (!asl.int<<1 : i32, [#asl.int_constraint<, 15 : i64>], 0 : i32>>, !asl.int<<0 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> !asl.int<<1 : i32, [#asl.int_constraint<, 15 : i64>], 0 : i32>>
// CHECK:   %12 = "asl.expr.literal.int"() <{value = "-1"}> : () -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %13 = "asl.expr.binop.mul"(%12, %3) : (!asl.int<<0 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %14 = "asl.expr.atc.int.range"(%arg1, %2, %13) : (!asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>, !asl.int<<0 : i32, [], 0 : i32>>) -> !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>
// CHECK:   %15 = "asl.expr.binop.plus"(%5, %4) : (!asl.int<<3 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:   %16 = "asl.expr.atc.int.range"(%arg2, %15, %6) : (!asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>
// CHECK:   %17 = "asl.expr.literal.int"() <{value = "0"}> : () -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %18 = "asl.expr.binop.eq"(%17, %7) : (!asl.int<<0 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> i1
// CHECK:   %19 = "asl.expr.literal.int"() <{value = "0"}> : () -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %20 = "asl.expr.binop.neq"(%19, %7) : (!asl.int<<0 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> i1
// CHECK:   %21 = "asl.expr.literal.int"() <{value = "0"}> : () -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %22 = "asl.expr.cond"(%20, %9, %21) : (i1, !asl.int<<3 : i32, [], 0 : i32>>, !asl.int<<0 : i32, [], 0 : i32>>) -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %23 = "asl.expr.cond"(%18, %8, %22) : (i1, !asl.int<<3 : i32, [], 0 : i32>>, !asl.int<<0 : i32, [], 0 : i32>>) -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %24 = "asl.expr.atc.int.exact"(%arg3, %23) : (!asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.int<<0 : i32, [], 0 : i32>>) -> !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>
// CHECK:   %25 = "asl.expr.literal.int"() <{value = "15"}> : () -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %26 = "asl.expr.call"(%25) <{call_type = 1 : i32, name = "Ones", params_size = 1 : i32}> : (!asl.int<<0 : i32, [], 0 : i32>>) -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:   %27 = "asl.expr.atc"(%26) <{target_type = !asl.bits<15 : i64, []>}> : (!asl.int<<0 : i32, [], 0 : i32>>) -> !asl.bits<15 : i64, []>
// CHECK:   "asl.stmt.return"(%27) : (!asl.bits<15 : i64, []>) -> ()
// CHECK: }) : () -> ()

func parameters_of_expressions{B, C, D, E, F, G, H, I, J}(
    w: integer{A..B},
    x: integer{C .. (- D)},
    y: integer{E+F .. (G)},
    z: integer{if H == 0 then I else J}) =>
    bits(A)
begin
    return Ones{A};
end;
