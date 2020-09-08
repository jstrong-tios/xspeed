#![feature(div_duration)]

#[macro_use]
extern crate slog;

use std::thread::{self, JoinHandle};
use std::time::*;
use std::str::FromStr;
use std::convert::TryFrom;
use rand::{thread_rng, Rng};
use rand::distributions::Standard;
use crossbeam_deque as deque;
use xgboost as xgb;
use slog::Drain;
use pretty_toa::ThousandsSep;


struct Worker {
    thread: Option<JoinHandle<()>>,
}

pub fn usize_validator(s: String) -> Result<(), String> {
    usize::from_str(&s).map_err(|e| {
        format!("failed to parse integer (usize): {} (input was '{}')", e, s)
    }).map(|_| ())
}

pub fn u32_validator(s: String) -> Result<(), String> {
    u32::from_str(&s).map_err(|e| {
        format!("failed to parse integer (u32): {} (input was '{}')", e, s)
    }).map(|_| ())
}

pub fn isize_validator(s: String) -> Result<(), String> {
    isize::from_str(&s).map_err(|e| {
        format!("failed to parse integer (isize): {} (input was '{}')", e, s)
    }).map(|_| ())
}

fn parse_n_threads(n: isize) -> Result<usize, String> {
    match n {
        -2              => Ok(num_cpus::get()),
        -1              => Ok(num_cpus::get_physical()),
        n if n >= 1     => Ok(n as usize),
        other           => Err(format!("Illegal value for number of threads: {}", other))
    }
}

pub fn per_sec(n: usize, span: Duration) -> f64 {
    if n == 0 || span < Duration::from_micros(1) { return 0.0 }
    Duration::from_secs(1)
        .div_duration_f64(span / u32::try_from(n).unwrap_or(std::u32::MAX))
}

impl Worker {
    pub fn thread(&mut self) -> Option<JoinHandle<()>> {
        self.thread.take()
    }

    pub fn new(
        id: usize,
        rx: deque::Stealer<Option<()>>,
        n_feat: usize,
        n_examples: usize,
        n_xgboost_threads: u32,
        n_boost_rounds: u32,
        n_trees: Option<u32>,
        logger: &slog::Logger,
    ) -> Self {
        let logger = logger.new(o!(
            "thread" => "Worker",
            "worker-id" => id,
            "n_feat" => n_feat,
            "n_examples" => n_examples,
            "n_xgboost_threads" => n_xgboost_threads));

        let thread = thread::spawn(move || {
            let start = Instant::now();
            info!(logger, "initializing");
            let mut rng = thread_rng();
            let size = n_feat * n_examples;
            let x: Vec<f32> = rng.sample_iter(&Standard).take(size).collect();
            let y: Vec<f32> = rng.sample_iter(&Standard).take(n_examples).collect();

            // xgb data ingest type
            let mut dmat = xgb::DMatrix::from_dense(&x[..], n_examples).expect("DMatrix::from_dense");
            dmat.set_labels(&y[..]).expect("DMatrix::set_labels");

            let tree_params = match n_trees {
                Some(n) => {
                    xgboost::parameters::tree::TreeBoosterParametersBuilder::default()
                        .tree_method(xgboost::parameters::tree::TreeMethod::Auto)
                        .num_parallel_tree(n)
                        .colsample_bynode(0.75)
                        .eta(1.0)
                        .max_depth(16)
                        .subsample(0.75)
                        .build()
                        .unwrap()
                }

                None => {
                    xgboost::parameters::tree::TreeBoosterParametersBuilder::default()
                        .build()
                        .unwrap()
                }
            };

            let booster_params = xgboost::parameters::BoosterParametersBuilder::default()
                .booster_type(xgboost::parameters::BoosterType::Tree(tree_params))
                .threads(Some(n_xgboost_threads))
                .build()
                .unwrap();

            let training_params = xgb::parameters::TrainingParametersBuilder::default()
                .dtrain(&dmat)
                .evaluation_sets(None)
                .booster_params(booster_params)
                .boost_rounds(n_boost_rounds)
                .build()
                .unwrap();

            let mut n_completed = 0;

            info!(logger, "ready for jobs");

            loop {
                match rx.steal() {
                    deque::Steal::Success(Some(_)) => {
                        let job_start = Instant::now();
                        let _ = xgb::Booster::train(&training_params).unwrap();
                        n_completed += 1;
                        info!(logger, "finished training in {:?}", Instant::now() - job_start; "n_completed" => n_completed);
                    }

                    deque::Steal::Empty | deque::Steal::Retry => {
                        trace!(logger, "no work to do");
                    }

                    deque::Steal::Success(None) => break, // termination command
                }
            }

            info!(logger, "terminating"; "took" => %format_args!("{:?}", Instant::now() - start));
        });

        Self {
            thread: Some(thread),
        }
    }
}

fn main() -> Result<(), String> {
    let start = Instant::now();
    let args: clap::ArgMatches = clap::App::new("xspeed")
        .version(clap::crate_version!())
        .arg(clap::Arg::with_name("n-features")
             .long("n-features")
             .short("f")
             .help("number of features (i.e. columns) for training data set")
             .default_value("25")
             .validator(usize_validator)
             .takes_value(true)
             .required(true))
        .arg(clap::Arg::with_name("n-examples")
             .long("n-examples")
             .short("e")
             .help("number of examples (i.e. rows) for training data set")
             .default_value("1000")
             .validator(usize_validator)
             .takes_value(true)
             .required(true))
        .arg(clap::Arg::with_name("n-threads")
             .long("n-threads")
             .short("j")
             .help("number of rust/os threads to split tasks by (-1 for number of physical cores, -2 for number logical cores)")
             .default_value("-1")
             .validator(isize_validator)
             .takes_value(true)
             .required(true))
        .arg(clap::Arg::with_name("n-xgboost-threads")
             .long("n-xgboost-threads")
             .short("x")
             .help("number of threads each xgboost model instructed to use (-1 for number of physical cores, -2 for number of logical cores)")
             .default_value("1")
             .validator(isize_validator)
             .takes_value(true)
             .required(true))
        .arg(clap::Arg::with_name("n-jobs")
             .long("n-jobs")
             .short("n")
             .help("number of models to train")
             .default_value("32")
             .validator(usize_validator)
             .takes_value(true)
             .required(true))
        .arg(clap::Arg::with_name("n-boost-rounds")
             .long("n-boost-rounds")
             .short("b")
             .help("number of boost rounds")
             .default_value("10")
             .validator(u32_validator)
             .takes_value(true)
             .required(true))
        .arg(clap::Arg::with_name("n-trees")
             .long("n-trees")
             .short("t")
             .help("random forest mode: number of trees each model will train with")
             .validator(u32_validator)
             .takes_value(true)
             .required(false))
        .get_matches();

    let decorator = slog_term::TermDecorator::new().stdout().force_color().build();
    let drain = slog_term::FullFormat::new(decorator).use_utc_timestamp().build().fuse();
    let drain = slog_async::Async::new(drain).chan_size(1024 * 64).thread_name("recv".into()).build().fuse();
    let logger = slog::Logger::root(drain, o!("version" => clap::crate_version!()));

    debug!(logger, "initializing");
    let n_feat = usize::from_str(args.value_of("n-features").unwrap()).unwrap(); // safe because input already validated with `validator(usize_validator)`
    let n_examples = usize::from_str(args.value_of("n-examples").unwrap()).unwrap();
    let n_threads = parse_n_threads(isize::from_str(args.value_of("n-threads").unwrap()).unwrap())?;
    let n_xgboost_threads = parse_n_threads(isize::from_str(args.value_of("n-xgboost-threads").unwrap()).unwrap())? as u32;
    let n_boost_rounds = u32::from_str(args.value_of("n-boost-rounds").unwrap()).unwrap();
    let n_trees: Option<u32> = args.value_of("n-trees").map(|s| u32::from_str(s).unwrap());
    let n_jobs = usize::from_str(args.value_of("n-jobs").unwrap()).unwrap();
    info!(logger, "parsed opts, beginning work";
        "n_feat" => n_feat, "n_examples" => n_examples, "n_jobs" => n_jobs,
        "n_threads" => n_threads, "n_xgboost_threads" => n_xgboost_threads);

    info!(logger, "launching workers"; "n_threads" => n_threads);
    let queue = deque::Worker::new_fifo();
    let mut workers = Vec::new();
    for i in 0..n_threads {
        workers.push(Worker::new(i, queue.stealer(), n_feat, n_examples, n_xgboost_threads, n_boost_rounds, n_trees, &logger));
    }

    info!(logger, "enqueueing jobs");
    for _ in 0..n_jobs {
        queue.push(Some(()));
    }

    info!(logger, "enqueueing termination orders at end of queue");
    for _ in 0..n_threads {
        queue.push(None); // at end of queue, tells workers to stop
    }

    info!(logger, "joining worker threads on completion");
    for (i, mut worker) in workers.drain(..).enumerate() {
        if let Some(thread) = worker.thread() {
            info!(logger, "joining worker {}", i);
            let _ = thread.join().unwrap();
            info!(logger, "joined worker {}", i);
        } else {
            panic!("expected worker to have thread");
        }
    }

    let took = Instant::now() - start;
    info!(logger, "finished in {:?}", Instant::now() - start;
        "jobs/sec" => %format_args!("{}", (per_sec(n_jobs, took).round() as i64).thousands_sep()),
        "per job" => %format_args!("{:?}", took / n_jobs as u32));

    Ok(())
}
