#set document(title: "Development Guide")

= Project Structure

== herdtools7

`herdtools7` is the official frontend for the Arm Architectural Specification Language (ASL) and serves as the source of truth for the language specification.

It is responsible to provide the OCaml library for ASL JSON backend; LRM pdf; and `aslref` as golden parsing reference. 

To build the Herdtools7, run:
```sh
nix build .#herdtools7 --out-link herdtools7
```

The `herdtools7` suite is utilized as a library within this project. The ASL Language Reference Manual (LRM), provided by this dependency, can be found at:
`asl-json-backend-out/share/doc/ASLReference.pdf`
`./herdtools7-out/bin/aslref`

== ASL JSON Backend

It is responsible for parsing and type-checking ASL code(using the library from herdtools7).
After type-checking, the resulting AST is converted to JSON.
Its design is intentionally minimal to reduce maintenance overhead.
The serialization results is consumed by ASL MLIR.

To build the ASL JSON Backend, run:
```sh
nix build .#asl-json-backend --out-link asl-json-backend-out
```

For development, use the following commands:
```sh
nix develop .#asl-json-backend
cd asl-json-backend
dune build
```

== ASL MLIR

The ASL MLIR component defines a one-to-one mapping of the ASL AST JSON to MLIR.
To reduce build times and ensure compatibility, the ASL MLIR dependency is aligned with the version used in the CIRCT project.

To build ASL MLIR, run:
```sh
nix build .#asl-mlir --out-link asl-mlir
```

For development, use the following commands to set up the build environment:
```sh
nix develop .#asl-mlir
cd mlir
cmake -G Ninja -B build
ninja -C build
```
