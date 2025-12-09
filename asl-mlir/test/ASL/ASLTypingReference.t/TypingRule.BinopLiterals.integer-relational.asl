// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
    println "eq_int: 5 == 10 = ", 5 == 10;
    println "ne_int: 5 != 10 = ", 5 != 10;
    println "le_int: 10 <= 10 = ", 10 <= 10;
    println "lt_int: 10 < 10 = ",  10 <= 10;
    println "lt_int: 5 < 10 = ",  5 <= 10;
    println "gt_int: 10 > 10 = ",  10 > 10;
    println "gt_int: 11 > 10 = ",  11 > 10;
    println "ge_int: 11 >= 10 = ",  11 >= 10;
    println "ge_int: 6 >= 10 = ",  6 >= 10;
    // println "invalid", 6 >= 0.0;
    return 0;
end;
