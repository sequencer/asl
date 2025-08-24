// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyException of exception{-};

func main() => integer
begin
    throw MyException{-};
    return 0;
end;
