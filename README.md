# xspeed

train some number of xgboost models on data of some shape with some number of threads and each xgboost model using some number of threads -- and measure how long it takes.

## usage

```console
$ ./target/release/xspeed -h
xspeed 0.1.0

USAGE:
    xspeed --n-examples <n-examples> --n-features <n-features> --n-jobs <n-jobs> --n-threads <n-threads> --n-xgboost-threads <n-xgboost-threads>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -e, --n-examples <n-examples>                  number of examples (i.e. rows) for training data set [default: 1000]
    -f, --n-features <n-features>                  number of features (i.e. columns) for training data set [default: 25]
    -n, --n-jobs <n-jobs>                          number of models to train [default: 32]
    -j, --n-threads <n-threads>
            number of rust/os threads to split tasks by (-1 for number of physical cores, -2 for number logical cores)
            [default: -1]
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

### run

```console
$ ./target/release/xspeed -j 8 -n 128
```

