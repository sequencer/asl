final: prev:

rec {
  herdtools7 = final.callPackage ./pkgs/herdtools7.nix { };

  circt-llvm = final.callPackage "${final.path}/pkgs/by-name/ci/circt/circt-llvm.nix" { };

  asl-llvm = circt-llvm.dev;

  asl-json-backend = final.callPackage ./pkgs/asl-json-backend.nix { };

  asl-mlir = final.callPackage ./pkgs/asl-mlir.nix { };
}
