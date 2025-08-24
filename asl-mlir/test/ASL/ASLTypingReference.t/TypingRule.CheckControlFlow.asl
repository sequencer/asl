// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

pure func Ones{N}() => bits(N)
begin
  return NOT Zeros{N};
end;

type invalid_state of exception{-};

func all_terminating_paths_correct{N}(v: bits(N), flag: boolean) => bits(N)
begin
    if v != Zeros{N} then
        if flag then
            return Ones{N} XOR v;
        else
            unreachable;
        end;
    else
        if flag then
            return v;
        else
            throw invalid_state{-};
        end;
    end;
end;
