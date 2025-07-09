{
  fetchFromGitHub,
  ocamlPackages,
  herdtools7,
  lit,
}:

ocamlPackages.buildDunePackage {
  pname = "asl-json-backend";
  src = ../../asl-json-backend;
  version = "0.1";

  nativeBuildInputs = with ocamlPackages; [
    ocaml
    dune_3
    yojson
    zarith
    menhirLib
    lit
  ];

  buildInputs = with ocamlPackages; [
    yojson
    zarith
    menhirLib
  ];

  configurePhase = ''
    runHook preConfigure
    # Disable dune cache to avoid permission issues
    export DUNE_CACHE=disabled
    export DUNE_CACHE_ROOT=/tmp/dune-cache-$$

    export OCAMLPATH=${herdtools7}/lib/:$OCAMLPATH
    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild

    dune build --profile release --cache disabled @install

    runHook postBuild
  '';

  doCheck = false;
  checkPhase = ''
    runHook preCheck
    dune build @lit
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp _build/default/src/JsonBackend.exe $out/bin/asl-json-backend

    runHook postInstall
  '';

  shellHook = ''
    export OCAMLPATH=${herdtools7}/lib/:$OCAMLPATH
  '';

  meta = {
    description = "ASL to MLIR converter";
    homepage = "https://github.com/sequencer/asl";
    license = "Apache";
    maintainers = [ ];
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };
}
