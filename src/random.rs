//! Random number generation utilities.
//!
//! **IMPORTANT**: This differs from original Walnut's implementation!
//! This crate only defines a way to generate random [`glm::Vector3`]s.
//! The static instance is not included, you must choose a random number generator
//! yourself that implements [`rand::Rng`], like `mt19937` from [`mersenne_twister` crate](https://docs.rs/mersenne_twister/latest/mersenne_twister/index.html)

use glm::Vector2;
use glm::Vector3;
use glm::Vector4;
use rand::distributions::{
    uniform::{SampleRange, SampleUniform},
    Distribution, Standard,
};

/// Enables generation of glm::VectorN instances
struct VecDistribution<D>(D);

impl<D> From<D> for VecDistribution<D> {
    fn from(dist: D) -> Self {
        Self(dist)
    }
}

impl<D, T> Distribution<Vector3<T>> for VecDistribution<D>
where
    D: Distribution<T>,
    T: glm::Primitive,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vector3<T> {
        let dist = &self.0;
        Vector3 {
            x: dist.sample(rng),
            y: dist.sample(rng),
            z: dist.sample(rng),
        }
    }
}

impl<D, T> Distribution<Vector2<T>> for VecDistribution<D>
where
    D: Distribution<T>,
    T: glm::Primitive,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vector2<T> {
        let dist = &self.0;
        Vector2 {
            x: dist.sample(rng),
            y: dist.sample(rng),
        }
    }
}

impl<D, T> Distribution<Vector4<T>> for VecDistribution<D>
where
    D: Distribution<T>,
    T: glm::Primitive,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vector4<T> {
        let dist = &self.0;
        Vector4 {
            x: dist.sample(rng),
            y: dist.sample(rng),
            z: dist.sample(rng),
            w: dist.sample(rng),
        }
    }
}

pub struct VecInRange<T>(std::ops::Range<T>);

impl<T> From<std::ops::Range<T>> for VecInRange<T> {
    fn from(r: std::ops::Range<T>) -> Self {
        Self(r)
    }
}

impl<T: glm::Primitive + PartialOrd + SampleUniform> SampleRange<Vector3<T>> for VecInRange<T> {
    fn sample_single<R: rand::RngCore + ?Sized>(self, rng: &mut R) -> Vector3<T> {
        Vector3 {
            x: self.0.clone().sample_single(rng),
            y: self.0.clone().sample_single(rng),
            z: self.0.sample_single(rng),
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T: glm::Primitive + PartialOrd + SampleUniform> SampleRange<Vector2<T>> for VecInRange<T> {
    fn sample_single<R: rand::RngCore + ?Sized>(self, rng: &mut R) -> Vector2<T> {
        Vector2 {
            x: self.0.clone().sample_single(rng),
            y: self.0.sample_single(rng),
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T: glm::Primitive + PartialOrd + SampleUniform> SampleRange<Vector4<T>> for VecInRange<T> {
    fn sample_single<R: rand::RngCore + ?Sized>(self, rng: &mut R) -> Vector4<T> {
        Vector4 {
            x: self.0.clone().sample_single(rng),
            y: self.0.clone().sample_single(rng),
            z: self.0.clone().sample_single(rng),
            w: self.0.sample_single(rng),
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

pub fn in_unit_sphere<R: rand::Rng>(rng: &mut R) -> glm::Vec3 {
    let dist = VecDistribution(Standard);
    unsafe {
        std::iter::from_fn(|| Some(dist.sample(rng)))
            .find(|v| glm::dot(*v, *v) < 1.0)
            .unwrap_unchecked()
    }
}
