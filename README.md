# uTensor Hello World repository
This is a quick end-to-end, training-to-deployment, uTensor demo. If you only wish to run the project, check the "Jump Start" section at the end of this doc.

## Requirements
In a Python virtual environment, install the following:
- mbed-cli
- utensor-cgen

*Please see the installation guide for mbed-cli and utensor-cli setup (to be linked).*

## End-to-end Instruction

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

## Jump Start
Alternately, if you are looking to just compile the project without getting into the training and code-generation, use the instructions below instead.
```
$ mbed import https://github.com/uTensor/utensor-helloworld
$ cd utensor-helloworld

# connect your board

$ mbed compile -m auto -t GCC_ARM -f
```