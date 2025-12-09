// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func println_me ()
begin
  for i = 0 to 42 do
    if i >= 3 then
      return;
    end;
  end;
  assert FALSE;

end;

func main () => integer
begin
    println_me ();

    return 0;
end;
