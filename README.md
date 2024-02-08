# Stack Overflow Error on Cairo Projects

This project is a Cairo representation of a Neural Netwrork. 

We encounter the following error when we compile the project: 
```bash 
$ cd inference
$ scarb build
>>>
thread 'main' has overflowed its stack
fatal runtime error: stack overflow
zsh: abort      scarb build
```

## Requirement
Scarb version 2.5.3