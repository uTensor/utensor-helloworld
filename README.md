Build Instruction:
```
mbed import https://github.com/uTensor/utensor-helloworld
cd utensor-helloworld
# connect your board
mbed compile -m auto -t GCC_ARM -f
```
Code Generation:
```
utensor-cli convert deep_mlp.pb --output-nodes=y_pred
```