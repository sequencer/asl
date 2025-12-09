// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

accessor f1() <=> v: integer
begin
  readonly getter
    return 3;
  end;

  setter
    assert v == 3;
  end;
end;

accessor f1b() <=> v: integer
begin
  readonly getter
    return 4;
  end;

  setter
    assert v == 4;
  end;
end;

accessor f2(x:integer) <=> v: integer
begin
  readonly getter
    return f1b() + x;
  end;

  setter
    f1b() = 4 * (v - x);
  end;
end;


accessor f3(x:integer) <=> v: integer
begin
  readonly getter
    return 0;
  end;

  setter
    assert x == 12;
    assert v == 13;
  end;
end;

func main() => integer
begin
  f1() = f1();
  // f1 = f1; // Illegal because f1 is not an empty setter/getter
  f1b() = f1b();
  let a = f1();
  assert a == 3;
  assert f1() == 3;
  let b = f1();
  assert b == 3;
  assert 3 == f1();
  let c = f2(4);
  assert c == 8;

  f2(5) = 6;
  f3(12) = 13;

  return 0;
end;

