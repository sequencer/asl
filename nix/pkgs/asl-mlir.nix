{
  stdenv,
  lib,
  cmake,
  coreutils,
  python3,
  git,
  fetchFromGitHub,
  ninja,
  lit,
  gitUpdater,
  callPackage,
  asl-llvm,
  asl-json-backend,
}:

let
  pythonEnv = python3.withPackages (ps: [ ps.psutil ]);
in
stdenv.mkDerivation rec {
  name = "asl-mlir";
  src = ../../asl-mlir;

  requiredSystemFeatures = [ "big-parallel" ];

  nativeBuildInputs = [
    cmake
    ninja
    git
    pythonEnv
  ];
  buildInputs = [
    asl-llvm
    asl-json-backend
    lit
  ];

  cmakeFlags = [
    "-DMLIR_DIR=${asl-llvm}/lib/cmake/mlir"
    # LLVM_EXTERNAL_LIT is executed by python3, the wrapped bash script will not work
    "-DLLVM_EXTERNAL_LIT=${lit}/bin/.lit-wrapped"
  ];

  doCheck = false;

  passthru = {
    llvm = asl-llvm;
  };

  meta = {
    description = "ASL Compiler";
    homepage = "https://github.com/sequencer/asl";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [
      sequencer
    ];
    platforms = lib.platforms.all;
  };
}
