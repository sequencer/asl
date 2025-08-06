#set document(title: "Development Guide")

= Environment Setup

The development environment is managed using Nix. To enter the development shell, run the following command from the project root:

```sh
nix develop
```

This command ensures all necessary dependencies are available. Some dependencies may be copied to the `reference/` directory for local access and AI agent use.

= Components

== herdtools7

`herdtools7` is the official frontend for the Arm Architectural Specification Language (ASL) and serves as the source of truth for the language specification. It is responsible for parsing and type-checking ASL code. After successful validation, the resulting Abstract Syntax Tree (AST) is consumed by the ASL JSON Backend.

To build the Herdtools7, run:
```sh
nix build .#herdtools7 --out-link herdtools7
```

The `herdtools7` suite is utilized as a library within this project. The ASL Language Reference Manual (LRM), provided by this dependency, can be found at:
`asl-json-backend-out/share/doc/ASLReference.pdf`
`./herdtools7-out/bin/aslref`

== ASL JSON Backend

This is a minimal tool for serializing the ASL AST, including type information, into JSON format. This serialization bridges the OCaml-based components with C++ tools. Its design is intentionally minimal to reduce maintenance overhead.

=== Building

To build the ASL JSON Backend, run:
```sh
nix build .#asl-json-backend --out-link asl-json-backend-out
```

=== Development

For development, use the following commands:
```sh
nix develop .#asl-json-backend
cd asl-json-backend
dune build
```

== ASL MLIR

The ASL MLIR component defines a one-to-one mapping of the ASL AST to MLIR. It also includes a conversion from the ASL dialect to the EmitC dialect, which enables the creation of an executable specification.

To reduce build times and ensure compatibility, the ASL MLIR dependency is aligned with the version used in the CIRCT project.

=== Building

To build ASL MLIR, run:
```sh
nix build .#asl-mlir --out-link asl-mlir
```

=== Development

For development, use the following commands to set up the build environment:
```sh
nix develop .#asl-mlir
cd mlir
cmake -G Ninja -B build
ninja -C build
```

= Roadmap
The project development roadmap is:
- Setup Herdtools7(aslib frontend) and dump the json AST;
- Prototype the MLIR Dialect;
- Parse all tests and convert them into MLIR;
- Add pass to lower the MLIR pass into EmitC Dialect;
- Compile T1 pokedex into native C for execution;
- Migrate more upstream tests into project and start to bump with the herdtools7 CI/CD; 
