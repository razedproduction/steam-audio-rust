//! Pathing: sound paths around obstacles through probes that see each other (upstream `path_visibility`,
//! `path_finder`, `path_data`).
//!
//! A [`PathBaker`] builds a [`ProbeVisibilityGraph`] over the probes of a batch, finds the shortest path between every
//! pair of probes with a [`PathFinder`], and stores the distinct ones as [`SoundPath`]s in a [`BakedPathData`] layer of
//! the batch. The path simulator and effect that use them follow in stage 7b.

mod baker;
mod data;
mod finder;
mod visibility;

pub use baker::PathBaker;
pub use data::{BakedPathData, PathBakeParams, SoundPath};
pub use finder::{PathFinder, ProbePath, ShortestPathParams};
pub use visibility::{
    AdjacencyEntry, ProbeVisibilityGraph, ProbeVisibilityTester, VisibilityParams,
};
