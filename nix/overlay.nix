final: prev:

rec {
  herdtools7 = final.callPackage ./pkgs/herdtools7.nix { };

  circt-llvm = final.callPackage "${final.path}/pkgs/by-name/ci/circt/circt-llvm.nix" { };

  asl-llvm = circt-llvm.dev;

  asl-mlir = final.callPackage ./pkgs/asl-mlir.nix { };
}
