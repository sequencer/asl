// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

accessor X() <=> value_in: bits (4)
begin
  readonly getter
        return '1000';
    end;
    setter
        unreachable;
    end;
end;
