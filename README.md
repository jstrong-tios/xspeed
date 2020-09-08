# xspeed

train some number of xgboost models on data of some shape with some number of threads and each xgboost model using some number of threads -- and measure how long it takes.

## usage

```console
$ ./target/release/xspeed -h
xspeed 0.2.0

USAGE:
    xspeed [OPTIONS] --n-boost-rounds <n-boost-rounds> --n-examples <n-examples> --n-features <n-features> --n-jobs <n-jobs> --n-threads <n-threads> --n-xgboost-threads <n-xgboost-threads>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --n-boost-rounds <n-boost-rounds>          number of boost rounds [default: 10]
    -e, --n-examples <n-examples>                  number of examples (i.e. rows) for training data set [default: 1000]
    -f, --n-features <n-features>                  number of features (i.e. columns) for training data set [default: 25]
    -n, --n-jobs <n-jobs>                          number of models to train [default: 32]
    -j, --n-threads <n-threads>
            number of rust/os threads to split tasks by (-1 for number of physical cores, -2 for number logical cores)
            [default: -1]
    -t, --n-trees <n-trees>                        random forest mode: number of trees each model will train with
    -x, --n-xgboost-threads <n-xgboost-threads>
            number of threads each xgboost model instructed to use (-1 for number of physical cores, -2 for number of
            logical cores) [default: 1]
```

## install

### install rust/cargo

rust installer from [https://rustup.rs/](rustup.rs):

```console
$ curl https://sh.rustup.rs -sSf | sh
```

### clone

```console
$ git clone git@github.com:jstrong-tios/xspeed.git
```

### build

```console
$ cd /path/to/xspeed
$ cargo build --bin --release
```

... or using justfile:

```console
$ just release-build
```

### run

```console
$ ./target/release/xspeed -j 8 -n 128
```

## random forest mode

in random forest mode, `--n-boost-rounds` should typically be set to `1`, and `--n-trees` is used to control the size of the model.

e.g.

```console
./target/release/xspeed --n-boost-rounds 1 --n-trees 500 -j 4 -x 1 -e 1000 -f 75 -n 16
```
