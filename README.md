## Jump Start
```
$ mbed import https://github.com/uTensor/utensor-helloworld
$ cd utensor-helloworld

# connect your board

$ mbed compile -m auto -t GCC_ARM -f
```

## End-to-end Instruction

Please see the installation guide for mbed-cli and utensor-cli setup (to be linked).

### Training
`$ python deep_mlp.py`

### Code Generation
```
utensor-cli convert deep_mlp.pb --output-nodes=y_pred
```

### Compile
```
$ mbed deploy
$ mbed compile -m auto -t GCC_ARM -f --sterm --baudrate=119200
```
Expected output:
`Predicted label: 7`