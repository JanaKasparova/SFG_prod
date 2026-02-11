{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313;
in
pkgs.mkShell {
  packages = [
    (python.withPackages (ps: [
      ps.aioftp
      ps.aiohttp
      ps.astropy
      ps.beautifulsoup4
      ps.colorlog
      ps.drms
      ps.h5netcdf
      ps.h5py
      ps.imageio
      ps.imutils
      ps.ipykernel
      ps.ipython
      ps.lxml
      ps.matplotlib
      ps.networkx
      ps.numpy
      ps.opencv4
      ps.pandas
      ps.parfive
      ps.pillow
      ps.psutil
      ps.pyerfa
      ps.pyzmq
      ps.requests
      ps.scikit-image
      ps.scipy
      ps.sunpy
      ps.tifffile
      ps.tqdm
      ps.zeep
    ]))
  ];
}

