{
  fetchgit,
  ocamlPackages,
  texliveFull,
  python3,
}:

ocamlPackages.buildDunePackage {
  pname = "herdtools7";
  version = "7.57+dev";

  src = fetchgit {
    url = "https://github.com/herd/herdtools7";
    rev = "d7d6bdd24f8680c4abf2df3a3e54a9d98494321e";
    hash = "sha256-Y2G6dPVnrVX1FewFZfcMCDsLqE7/N0qZxTuoNupz3rU=";
  };

  nativeBuildInputs =
    with ocamlPackages;
    [
      ocaml
      dune_3
      menhir
      zarith
      menhirLib
    ]
    ++ [
      texliveFull
      python3
    ];

  buildInputs = with ocamlPackages; [
    findlib
    zarith
    menhirLib
    qcheck
    menhirSdk
    logs
  ];

  configurePhase = ''
    runHook preConfigure

    # Patch
    rm asllib/carpenter asllib/menhir2bnfc -rf
    sed -i 's/\\\\\\\\/\\\\/g' asllib/doc/Makefile

    # Disable dune cache to avoid permission issues
    export DUNE_CACHE=disabled
    export DUNE_CACHE_ROOT=/tmp/dune-cache-$$


    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild

    # Generate version information in the project root
    bash ./version-gen.sh $out

    # Disable dune cache to avoid permission issues
    export DUNE_CACHE=disabled

    dune build --profile release --cache disabled @install

    # Build the ASL Reference PDF
    pushd asllib/doc
    make ASLReference.pdf
    popd

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    dune install --prefix=$out

    # Install the ASL Reference PDF
    mkdir -p $out/doc
    cp asllib/doc/ASLReference.pdf $out/doc/

    runHook postInstall
  '';

  meta = {
    description = "ASL reference implementation and library";
    homepage = "https://github.com/herd/herdtools7";
    license = "BSD-2-Clause";
    maintainers = [ ];
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };
}
