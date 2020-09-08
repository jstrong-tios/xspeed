
export RUSTFLAGS := "-C target-cpu=native"
export MAKEFLAGS := "-j8"

cargo +args='':
    cargo {{args}}

check +args='':
    @just cargo check {{args}}

release-build +args='':
    cargo build --release --bin xspeed {{args}}


