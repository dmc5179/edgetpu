# Edge TPU Python API

This repository contains an easy-to-use Python API to work with Coral devices:

* [Dev Board](https://coral.withgoogle.com/products/dev-board/)
* [USB Accelerator](https://coral.withgoogle.com/products/accelerator/)

You can run inference and do transfer learning.


## Build and install from source

1. Sync the source code as per the [Mendel get started guide](
https://coral.googlesource.com/docs/+/refs/heads/master/GettingStarted.md).

1. `cd packages/edgetpu/`

1. `./build_package.sh`

1. `tar xzf edgetpu_api_<version>.tar.gz`

1. `cd edgetpu_api/`

1. `sudo ./install.sh`

1. Now check for the installed library at `/usr/local/lib/python3.6/dist-packages/edgetpu/`

If it seems the library did not update compared to an older version you had installed, then
run the `uninstall.sh` script and then rerun `install.sh` (it probably skipped installing the new
library because the version number hasn't been updated yet).
