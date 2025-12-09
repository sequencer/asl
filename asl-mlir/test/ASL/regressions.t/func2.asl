// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

accessor X(i:integer) <=> v: integer
begin
  readonly getter
      return i;
  end;

  setter
      let internal_i = i;
      let internal_v = v;
  end;
end;

func main() => integer
begin
    X(2) = 3;
    let x = X(4);

    assert x == 4;

    return 0;
end;

