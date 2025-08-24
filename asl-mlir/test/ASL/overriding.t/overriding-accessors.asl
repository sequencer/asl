// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var state : integer = 0;
var called_override = FALSE;

impdef accessor Foo() <=> v: integer
begin
  getter
    return state;
  end;

  setter
    state = v;
  end;
end;

implementation accessor Foo() <=> v: integer
begin
  getter
    called_override = TRUE;
    return state;
  end;

  setter
    state = 42;
  end;
end;

func main() => integer
begin
  - = Foo();
  assert called_override;
  Foo() = 1;
  assert state == 42;
  return 0;
end;
