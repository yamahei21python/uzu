{
  pkgs,
  native,
  sources,
}:
native {
  name = "engine-llamacpp";
  src = pkgs.lib.fileset.toSource {
    root = ../.;
    fileset = pkgs.lib.fileset.unions [./CMakeLists.txt ./src ../common-cpp ../sources.json];
  };
  sourceRoot = "source/engine-llamacpp";
  nativeBuildInputs = [pkgs.cmake];
  cmakeFlags = [
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=${sources.llama_cpp}"
    "-DFETCHCONTENT_SOURCE_DIR_CLI11=${sources.cli11}"
    "-DFETCHCONTENT_SOURCE_DIR_GLAZE=${sources.glaze}"
    "-DGGML_NATIVE=OFF"
  ];
  installPhase = ''install -Dm755 engine-llamacpp "$out/bin/engine-llamacpp"'';
}
