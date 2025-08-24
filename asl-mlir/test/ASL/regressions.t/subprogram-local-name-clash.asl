// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

accessor X() <=> v: integer
begin
  readonly getter
    return 0;
  end;

  setter
    pass;
  end;
end;

func main() => integer
begin
  assert X() == 0;
  let X = 1;
  X() = 2;

  return 0;
end;
