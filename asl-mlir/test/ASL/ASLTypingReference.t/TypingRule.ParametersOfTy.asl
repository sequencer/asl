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

// CHECK:   "asl.func"() <{args = ["x", "y", "z", "w"], args_types = [!asl.tuple<[!asl.bits<-1 : i64, []>, !asl.bits<-1 : i64, []>]>, !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.real, !asl.int<<0 : i32, [], 0 : i32>>], builtin = false, name = "parameters_of_types", parameters = [{identifier = "A", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "B", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "C", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "D", type = !asl.int<<3 : i32, [], 0 : i32>>}, {identifier = "E", type = !asl.int<<3 : i32, [], 0 : i32>>}], primitive = false, return_type = !asl.bits<-1 : i64, []>, subprogram_type = 1 : i32}> ({
// CHECK:   ^bb0(%arg0: !asl.tuple<[!asl.bits<-1 : i64, []>, !asl.bits<-1 : i64, []>]>, %arg1: !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, %arg2: !asl.real, %arg3: !asl.int<<0 : i32, [], 0 : i32>>):
// CHECK:     %0 = "asl.expr.var"() <{name = "A"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:     %1 = "asl.expr.var"() <{name = "B"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:     %2 = "asl.expr.var"() <{name = "C"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:     %3 = "asl.expr.var"() <{name = "D"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:     %4 = "asl.expr.var"() <{name = "E"}> : () -> !asl.int<<3 : i32, [], 0 : i32>>
// CHECK:     %5 = "asl.expr.atc"(%arg0) <{target_type = !asl.tuple<[!asl.bits<-1 : i64, []>, !asl.bits<-1 : i64, []>]>}> : (!asl.tuple<[!asl.bits<-1 : i64, []>, !asl.bits<-1 : i64, []>]>) -> !asl.tuple<[!asl.bits<-1 : i64, []>, !asl.bits<-1 : i64, []>]>
// CHECK:     %6 = "asl.expr.atc.int.range"(%arg1, %3, %4) : (!asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> !asl.int<<1 : i32, [#asl.int_constraint<>], 0 : i32>>
// CHECK:     %7 = "asl.expr.atc"(%arg2) <{target_type = !asl.real}> : (!asl.real) -> !asl.real
// CHECK:     %8 = "asl.expr.atc"(%arg3) <{target_type = !asl.int<<0 : i32, [], 0 : i32>>}> : (!asl.int<<0 : i32, [], 0 : i32>>) -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:     %9 = "asl.expr.call"(%0) <{call_type = 1 : i32, name = "Ones", params_size = 1 : i32}> : (!asl.int<<3 : i32, [], 0 : i32>>) -> !asl.int<<0 : i32, [], 0 : i32>>
// CHECK:     %10 = "asl.expr.atc.bits"(%9, %0) <{bitfields = []}> : (!asl.int<<0 : i32, [], 0 : i32>>, !asl.int<<3 : i32, [], 0 : i32>>) -> !asl.bits<-1 : i64, []>
// CHECK:     "asl.stmt.return"(%10) : (!asl.bits<-1 : i64, []>) -> ()
// CHECK:   }) : () -> ()


func parameters_of_types{A, B, C, D, E}(
    x: (bits(B), bits(C)),
    y: integer{D..E},
    z: real,
    w: integer) =>
    bits(A)
begin
    return Ones{A};
end;
