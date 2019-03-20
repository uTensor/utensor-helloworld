Build Instruction:
```
mbed import https://github.com/uTensor/utensor-helloworld
cd utensor-helloworld
# connect your board
mbed compile -m auto -t GCC_ARM --profile=uTensor/build_profile/release.json -f
```