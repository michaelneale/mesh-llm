mod cli;
mod report;
mod runner;
mod support;

use anyhow::Result;
use clap::Parser;

use crate::{
    cli::{Cli, CommandKind},
    runner::{chain, dtype_matrix, single_step, split_scan},
};

fn main() -> Result<()> {
    match Cli::parse().command {
        CommandKind::SingleStep(args) => single_step(args),
        CommandKind::Chain(args) => chain(args),
        CommandKind::SplitScan(args) => split_scan(args),
        CommandKind::DtypeMatrix(args) => dtype_matrix(args),
    }
}
