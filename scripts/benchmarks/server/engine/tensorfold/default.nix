{
  pkgs,
  native,
  sources,
  server,
}: let
  engine = native {
    name = "tensorfold-native";
    src = sources.tensorfold;
    patches = [./patches/sdk-paths.patch];
    nativeBuildInputs = [pkgs.zig_0_17];
    zigBuildFlags = [
      "native"
      "-Doptimize=ReleaseFast"
      "-Dmetal-std=metal4.0"
      "-Dtarget=aarch64-macos.26.2"
    ];
    installPhase = ''install -Dm755 zig-out/native/bin/tensorfold-native "$out/bin/tensorfold-native"'';
  };
in
  server "tensorfold" {BENCHMARK_TENSORFOLD_BINARY = "${engine}/bin/tensorfold-native";}
