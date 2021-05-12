"""Script which downloads the build dependencies."""

import argparse
import os
import shutil
import subprocess
import zipfile

import requests


EIGEN = "https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip"
GFLAGS = "https://github.com/gflags/gflags/archive/v2.2.2.zip"
CERES = "https://github.com/ceres-solver/ceres-solver/archive/2.0.0.zip"
GLOG = "https://github.com/google/glog/archive/v0.4.0.zip"


def _download_and_unzip(download_uri, name):
    libs_dir = os.path.dirname(__file__)
    assert download_uri.endswith(".zip")
    zip_path = os.path.join(libs_dir, name + ".zip")

    if not os.path.exists(zip_path):
        print("Downloading", name, "from", download_uri)
        response = requests.get(download_uri)
        with open(zip_path, "wb") as file:
            file.write(response.content)

    print("Extracting", name)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(libs_dir)

    print("Done.")


def _build_dep(source_dir, install_dir, config, extra_args=None, target_dir="..", build_dir="build"):
    libs_dir = os.path.dirname(__file__)
    build_dir = build_dir + "_" + config.lower()
    build_dir = os.path.join(libs_dir, source_dir, build_dir)
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    install_dir = os.path.join(libs_dir, config, install_dir)
    install_dir = os.path.abspath(install_dir)
    config_args = [
        "cmake.exe", target_dir,
        "-DCMAKE_INSTALL_PREFIX=" + install_dir
    ]

    if extra_args:
        config_args.extend(extra_args)

    subprocess.check_call(config_args, cwd=build_dir)
    subprocess.check_call([
        "cmake.exe",
        "--build", ".",
        "--config", config,
        "--target", "INSTALL"
    ], cwd=build_dir)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    libs_dir = os.path.dirname(__file__)
    libs_dir = os.path.abspath(libs_dir)

    if not os.path.exists(os.path.join(libs_dir, args.config, "Eigen3", "include")):
        _download_and_unzip(EIGEN, "eigen")
        _build_dep("eigen-3.3.9", "Eigen3", args.config)

    if not os.path.exists(os.path.join(libs_dir, args.config, "gflags", "include")):
        _download_and_unzip(GFLAGS, "gflags")
        _build_dep("gflags-2.2.2", "gflags", args.config)

    if not os.path.exists(os.path.join(libs_dir, args.config, "glog", "include")):
        _download_and_unzip(GLOG, "glog")
        _build_dep("glog-0.4.0", "glog", args.config, ["-DWITH_GFLAGS=0"])

    gflags_DIR = os.path.join(libs_dir, args.config, "gflags")
    glog_DIR = os.path.join(libs_dir, args.config, "glog")
    if not os.path.exists(os.path.join(libs_dir, args.config, "ceres", "include")):
        _download_and_unzip(CERES, "ceres")
        shutil.copy(os.path.join(libs_dir, "..", "cmake", "FindLAPACK.cmake"),
                    os.path.join(libs_dir, "ceres-solver-2.0.0", "cmake", "FindLAPACK.cmake"))
        extra_args = [
            "-DBUILD_TESTING=0",
            "-DBUILD_EXAMPLES=0",
            "-Dgflags_DIR={}".format(gflags_DIR),
            "-Dglog_DIR={}".format(glog_DIR),
            "-DCMAKE_PREFIX_PATH=../cmake"
        ]
        _build_dep("ceres-solver-2.0.0", "ceres", args.config, extra_args)


if __name__ == "__main__":
    _main()
