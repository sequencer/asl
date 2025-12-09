// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
    // local storage declaration    Side effect descriptors from keyword
    let l = 20;                     // LocalEffect(SE_Readonly), Immutability(TRUE)
    var v = 30;                     // LocalEffect(SE_Readonly), Immutability(FALSE)
    return 0;
end;
