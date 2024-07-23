{
  description = "flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = "${system}";
        config.allowUnfree = true;
      };
      opencvGtk = pkgs.opencv.override (old : { enableGtk2 = true; });
      dependencies = [
        pkgs.cmake
        pkgs.gcc
        pkgs.ncnn
        opencvGtk
        pkgs.vulkan-headers
        pkgs.vulkan-loader
        pkgs.vulkan-tools
        pkgs.ffmpeg_6-full
        pkgs.vulkan-validation-layers
        (pkgs.python311.withPackages (p: with p; [
          torchvision
          pytorch
        ]))
      ];
      libPath = pkgs.lib.makeLibraryPath dependencies;
      binPath = pkgs.lib.makeBinPath dependencies;
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = dependencies;
        shellHook = ''
            export VULKAN_SDK=${pkgs.vulkan-loader}
            export VULKAN_INCLUDE_DIR=${pkgs.vulkan-headers}/include
            export VULKAN_LIBRARY=${pkgs.vulkan-loader}/lib/libvulkan.so
            export NCNN_INSTALL_DIR=${pkgs.ncnn}
            export OpenCV_DIR=${opencvGtk}
          '';
      };
    };
}
