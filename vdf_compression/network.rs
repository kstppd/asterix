/* network.rs: A small MLP written in Rust for prototyping purposes
* Author: Kostis Papadakis, 2024 (kpapadakis@protonmail.com)

* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
* USA.
* */
#![allow(dead_code)]
pub mod network {

    use nalgebra::DMatrix;
    use num_traits::{Float, FromPrimitive, ToPrimitive};
    use rand::prelude::SliceRandom;
    use rand::{distributions::Uniform, Rng};
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;
    use std::convert::From;
    use std::convert::TryInto;
    use std::fs::File;
    use std::io::{Read, Write};
    

    //Constants used in ADAM and ADAMW optmizers
    const ADAM_BETA1: f32 = 0.9;
    const ADAM_BETA2: f32 = 0.999;
    const ADAM_EPS: f32 = 1e-8;
    const ADAM_DECAY: f32 = 0.01;
    const THREADS: usize = 8; //fallback value

    //TAG used to fingerprint state files
    const TAG: &str = "_TINYAI_";

    trait FromNetworkBytes {
        fn from_ne_bytes(a: &mut [u8]) -> Self;
    }

    impl<const N: usize> FromNetworkBytes for [u8; N] {
        fn from_ne_bytes(a: &mut [u8]) -> [u8; N] {
            let mut retval = [0u8; N];
            retval.copy_from_slice(a);
            retval
        }
    }

    impl FromNetworkBytes for u64 {
        fn from_ne_bytes(a: &mut [u8]) -> u64 {
            u64::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    impl FromNetworkBytes for u32 {
        fn from_ne_bytes(a: &mut [u8]) -> u32 {
            u32::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    impl FromNetworkBytes for f32 {
        fn from_ne_bytes(a: &mut [u8]) -> f32 {
            f32::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    impl FromNetworkBytes for f64 {
        fn from_ne_bytes(a: &mut [u8]) -> f64 {
            f64::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    impl FromNetworkBytes for usize {
        fn from_ne_bytes(a: &mut [u8]) -> usize {
            usize::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    impl FromNetworkBytes for u8 {
        fn from_ne_bytes(a: &mut [u8]) -> u8 {
            u8::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    impl FromNetworkBytes for u16 {
        fn from_ne_bytes(a: &mut [u8]) -> u16 {
            u16::from_ne_bytes(FromNetworkBytes::from_ne_bytes(a))
        }
    }

    fn parse_bytes<T: FromNetworkBytes + num_traits::Num>(input: &mut [u8]) -> Vec<T> {
        let elem_size = std::mem::size_of::<T>();
        let output = input
            .chunks_mut(elem_size)
            .map(|x| T::from_ne_bytes(x))
            .collect();
        output
    }

    //Yeah we unfirtunately need this
    fn clamp<T: Float>(val: T, min: T, max: T) -> T {
        if val < min {
            min
        } else if val > max {
            max
        } else {
            val
        }
    }

    fn quantize_vector<T: Float + ToPrimitive>(
        input_vector: &[T],
        bits: i32,
    ) -> (T, T, Vec<usize>) {
        assert!(!input_vector.is_empty());
        let max_val = *input_vector
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let min_val = *input_vector
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let num_levels = T::from(1u32 << bits).unwrap();
        let delta = (max_val - min_val) / (num_levels - T::one());
        let scale = T::one() / delta;

        let mut kappa: Vec<usize> = Vec::with_capacity(input_vector.len());
        for val in input_vector.iter() {
            let c: T = clamp(*val, min_val, max_val);
            let q = ((c - min_val) * scale + T::from(0.5).unwrap())
                .to_usize()
                .unwrap();
            kappa.push(q);
        }
        (delta, min_val, kappa)
    }

    fn dequantize_vector<T: Float + FromPrimitive>(
        delta: T,
        min_val: T,
        kappa: &[usize],
    ) -> Vec<T> {
        let mut output_vector: Vec<T> = Vec::with_capacity(kappa.len());
        for &q in kappa.iter() {
            let val = T::from_usize(q).unwrap() * delta + min_val;
            output_vector.push(val);
        }
        output_vector
    }

    fn pack_quantization_indices(quantized_weights: &[usize], bits: i32) -> Vec<u8> {
        let packed_quantized_weights: Vec<u8> = match bits {
            16 => quantized_weights
                .iter()
                .map(|x| u16::from_usize(*x).unwrap())
                .collect::<Vec<u16>>()
                .clone()
                .iter()
                .map(|x| x.to_ne_bytes())
                .collect::<Vec<[u8; 2]>>()
                .clone()
                .into_iter()
                .flatten()
                .collect(),
            8 => quantized_weights
                .iter()
                .map(|x| u8::from_usize(*x).unwrap())
                .collect(),
            _ => panic!("Quantization level not supported."),
        };
        packed_quantized_weights
    }

    fn unpack_quantization_indices<
        T: FromNetworkBytes + FromPrimitive + num_traits::Num + Float,
    >(
        buffer: &mut [u8],
        delta: T,
        min_val: T,
        bits: i32,
    ) -> Vec<T> {
        let weights: Vec<T> = match bits {
            0 => parse_bytes(buffer),
            8 => {
                let kappa_packed: Vec<u8> = parse_bytes(buffer);
                let kappa: Vec<usize> = kappa_packed
                    .iter()
                    .map(|x| usize::from_u8(*x).unwrap())
                    .collect();
                let w: Vec<T> = dequantize_vector(delta, min_val, &kappa);
                w
            }
            16 => {
                let kappa_packed: Vec<u16> = parse_bytes(buffer);
                let kappa: Vec<usize> = kappa_packed
                    .iter()
                    .map(|x| usize::from_u16(*x).unwrap())
                    .collect();
                let w: Vec<T> = dequantize_vector(delta, min_val, &kappa);
                w
            }
            _ => panic!("Selected Quantization level is not supported!"),
        };
        weights
    }

    trait ToBytes {
        fn to_bytes(self) -> Vec<u8>;
    }

    impl ToBytes for f32 {
        fn to_bytes(self) -> Vec<u8> {
            self.to_le_bytes().to_vec()
        }
    }

    impl ToBytes for f64 {
        fn to_bytes(self) -> Vec<u8> {
            self.to_le_bytes().to_vec()
        }
    }

    impl ToBytes for usize {
        fn to_bytes(self) -> Vec<u8> {
            self.to_le_bytes().to_vec()
        }
    }

    fn vec_to_bytes<T: ToBytes + Copy>(input: &[T]) -> Vec<u8> {
        let mut byte_vec: Vec<u8> = Vec::with_capacity(std::mem::size_of_val(input));
        for &val in input.iter() {
            byte_vec.extend(val.to_bytes());
        }
        byte_vec
    }

    pub trait NeuralNetworkTrait<T>:
        nalgebra::RealField
        + num_traits::Num
        + From<f32>
        + rand::distributions::uniform::SampleUniform
        + std::marker::Copy
        + 'static
        + Sized
        + num_traits::FromBytes
    {
    }

    impl<T, U> NeuralNetworkTrait<U> for T where
        T: nalgebra::RealField
            + num_traits::Num
            + From<f32>
            + rand::distributions::uniform::SampleUniform
            + std::marker::Copy
            + 'static
            + Sized
            + num_traits::FromBytes
    {
    }

    #[derive(Debug, Clone)]
    struct FullyConnectedLayer<T> {
        /*
            w->  weights
            b->  bias
            z->  result of Ax+b
            a->  result of act(z)
            delta -> used in backprop
            dw-> weights' gradients
            db-> biases' gradients
            opt_v-> weight velocity
            opt_m-> weight momentum
        */
        w: DMatrix<T>,
        b: DMatrix<T>,
        b_broadcasted: DMatrix<T>,
        z: DMatrix<T>,
        a: DMatrix<T>,
        dw: DMatrix<T>,
        db: DMatrix<T>,
        delta: DMatrix<T>,
        v: DMatrix<T>,
        m: DMatrix<T>,
        neurons: usize,
        batch_size: usize,
    }

    impl<T> FullyConnectedLayer<T>
    where
        T: NeuralNetworkTrait<T>,
    {
        pub fn new(neurons: usize, input_size: usize, batch_size: usize) -> Self {
            let mut rng = rand::thread_rng();
            let range1 = Uniform::<T>::new(-Into::<T>::into(0.5), Into::<T>::into(0.5));
            let range2 = Uniform::<T>::new(-Into::<T>::into(0.5), Into::<T>::into(0.5));
            FullyConnectedLayer {
                w: DMatrix::from_fn(input_size, neurons, |_, _| rng.sample(&range1)),
                b: DMatrix::from_fn(1, neurons, |_, _| rng.sample(&range2)),
                b_broadcasted: DMatrix::<T>::zeros(batch_size, neurons),
                z: DMatrix::<T>::zeros(batch_size, neurons),
                a: DMatrix::<T>::zeros(batch_size, neurons),
                dw: DMatrix::<T>::zeros(input_size, neurons),
                db: DMatrix::<T>::zeros(1, neurons),
                delta: DMatrix::<T>::zeros(batch_size, neurons),
                v: DMatrix::<T>::zeros(input_size, neurons),
                m: DMatrix::<T>::zeros(input_size, neurons),
                neurons,
                batch_size,
            }
        }

        fn total_bytes(&self) -> usize {
            let mut total_size: usize = 0;
            total_size += self.w.ncols() * self.w.nrows();
            total_size += self.b.ncols() * self.b.nrows();
            total_size * std::mem::size_of::<T>()
        }

        pub fn reset_deltas(&mut self) {
            self.dw.fill(T::zero());
            self.db.fill(T::zero());
            self.delta.fill(T::zero());
        }

        pub fn elu(val: T) -> T {
            if val >= T::zero() {
                val
            } else {
                let alpha: T = T::one();
                alpha * (val.exp() - T::one())
            }
        }

        pub fn elu_mut(val: &mut T) {
            if *val >= T::zero() {
            } else {
                let alpha: T = T::one();
                *val = alpha * ((*val).exp() - T::one())
            }
        }

        fn elu_prime(val: T) -> T {
            let alpha: T = T::one();
            if val >= T::zero() {
                T::one()
            } else {
                Self::elu(val) + alpha
            }
        }

        fn elu_prime_mut(val: &mut T) {
            let alpha: T = T::one();
            if *val >= T::zero() {
                *val = T::one();
            } else {
                *val = Self::elu(*val) + alpha;
            }
        }

        fn tanh_mut(val: &mut T) {
            let ex: T = T::exp(*val);
            let exn: T = T::exp(-*val);
            *val = (ex - exn) / (ex + exn);
        }

        fn tanh(val: T) -> T {
            let ex: T = T::exp(val);
            let exn: T = T::exp(-val);
            (ex - exn) / (ex + exn)
        }

        fn tanh_prime_mut(val: &mut T) {
            let tanh_val = Self::tanh(*val);
            *val = T::one() - tanh_val * tanh_val;
        }

        fn tanh_prime(val: T) -> T {
            let tanh_val = Self::tanh(val);
            T::one() - tanh_val * tanh_val
        }

        fn leaky_relu_mut(val: &mut T) {
            *val = if *val > T::zero() { *val } else { 0.001.into() };
        }

        fn leaky_relu(val: T) -> T {
            if val > T::zero() {
                val
            } else {
                0.001.into()
            }
        }

        fn leaky_relu_prime_mut(val: &mut T) {
            if *val > T::zero() {
                *val = T::one();
            } else {
                *val = 0.001.into();
            }
        }

        fn leaky_relu_prime(val: T) -> T {
            if val > T::zero() {
                T::one()
            } else {
                0.001.into()
            }
        }
        pub fn sigmoid(val: T) -> T {
            T::one() / (T::one() + (-val).exp())
        }

        pub fn sigmoid_mut(val: &mut T) {
            *val = T::one() / (T::one() + (-*val).exp());
        }

        fn sigmoid_prime(val: T) -> T {
            let sigmoid_val = Self::sigmoid(val);
            sigmoid_val * (T::one() - sigmoid_val)
        }

        fn sigmoid_prime_mut(val: &mut T) {
            let sigmoid_val = Self::sigmoid(*val);
            *val = sigmoid_val * (T::one() - sigmoid_val);
        }
        pub fn softmax(val: T) -> T {
            val.exp() // Softmax for a single value is just the exponential function
        }

        pub fn softmax_mut(val: &mut T) {
            *val = val.exp(); // Softmax for a single value is just the exponential function
        }

        pub fn softmax_prime(val: T) -> T {
            let softmax_val = Self::softmax(val);
            softmax_val * (T::one() - softmax_val) // Derivative of softmax function
        }

        pub fn softmax_prime_mut(val: &mut T) {
            let softmax_val = Self::softmax(*val);
            *val = softmax_val * (T::one() - softmax_val); // Derivative of softmax function
        }

        pub fn activate(val: T) -> T {
            FullyConnectedLayer::elu(val)
        }
        pub fn activate_mut(val: &mut T) {
            FullyConnectedLayer::elu_mut(val);
        }

        pub fn activate_prime(val: T) -> T {
            FullyConnectedLayer::elu_prime(val)
        }

        pub fn activate_prime_mut(val: &mut T) {
            FullyConnectedLayer::elu_prime_mut(val);
        }

        pub fn activate_prime_matrix(matrix: &DMatrix<T>) -> DMatrix<T> {
            matrix.map(|x| Self::activate_prime(x))
        }

        fn broadcast_biases(&mut self) {
            // b: DMatrix::from_fn(1, neurons, |_, _| rng.sample(&range2)),
            // b_broadcasted: DMatrix::from_fn(batch_size, neurons, |_, _| r
            for i in 0..self.batch_size {
                for j in 0..self.b.ncols() {
                    self.b_broadcasted[(i, j)] = self.b[(0, j)];
                }
            }
        }

        pub fn forward_matrix(&mut self, input: &DMatrix<T>) {
            self.broadcast_biases();
            input.mul_to(&self.w, &mut self.z);
            self.z += &self.b_broadcasted;
            self.a = self.z.clone();
            self.a.apply(|x| Self::activate_mut(x));
        }
    }

    #[derive(Debug, Clone)]
    pub struct Network<T> {
        layers: Vec<FullyConnectedLayer<T>>,
        m_thread_layers: Vec<Vec<FullyConnectedLayer<T>>>,
        input_data: DMatrix<T>,
        output_data: DMatrix<T>,
        batch_size: usize,
        neurons_per_layer: Vec<usize>,
        // input_buffer :mut DMatrix::<T>,
        // output_buffer :mut DMatrix::<T>
    }

    impl<T> Network<T>
    where
        T: NeuralNetworkTrait<T> + for<'a> std::iter::Sum<&'a T>,
    {
        pub fn new(
            input_size: usize,
            output_size: usize,
            hidden_layers: Vec<usize>,
            input: &DMatrix<T>,
            output: &DMatrix<T>,
            batch_size: usize,
        ) -> Self {
            let mut v: Vec<FullyConnectedLayer<T>> = vec![];
            let thread_layers_object: Vec<Vec<FullyConnectedLayer<T>>> = vec![];
            v.push(FullyConnectedLayer::new(
                *hidden_layers.first().unwrap(),
                input_size,
                batch_size,
            ));
            for i in 1..hidden_layers.len() {
                v.push(FullyConnectedLayer::new(
                    hidden_layers[i],
                    hidden_layers[i - 1],
                    batch_size,
                ));
            }
            v.push(FullyConnectedLayer::new(
                output_size,
                *hidden_layers.last().unwrap(),
                batch_size,
            ));
            assert_eq!(input_size, input.ncols());
            assert_eq!(output_size, output.ncols());
            Network {
                layers: v,
                m_thread_layers: thread_layers_object,
                input_data: input.clone(),
                output_data: output.clone(),
                batch_size,
                neurons_per_layer: hidden_layers,
            }
        }

        pub fn new_from_other_with_batchsize(other: &Network<T>, batch_size: usize) -> Self {
            let mut net = Network::<T>::new(
                other.input_data.ncols(),
                other.output_data.ncols(),
                other.neurons_per_layer.clone(),
                &other.input_data,
                &other.output_data,
                batch_size,
            );
            //Copy bias and weights. This way we have a new network with a different batch size but same latent space
            for i in 0..net.layers.len() {
                net.layers[i].w = other.layers[i].w.clone();
                net.layers[i].b = other.layers[i].b.clone();
            }
            net
        }

        pub fn randomize_he(&mut self) {
            let mut rng = rand::thread_rng();
            for i in 1..self.layers.len() {
                let fan_in = self.layers[i - 1].w.len();
                let he_stddev: f32 = (2.0 / (fan_in as f32)).sqrt();
                let normal = Normal::new(0.0, he_stddev).unwrap();
                let mut he = |elem: &mut T| {
                    *elem = T::from(normal.sample(&mut rng));
                };

                self.layers[i].w.apply(&mut he);
                self.layers[i].b.apply(he);
            }
        }

        pub fn randomize_xavier(&mut self) {
            let mut rng = rand::thread_rng();
            for i in 1..self.layers.len() {
                let fan_in = if i > 0 { self.layers[i - 1].w.len() } else { 0 };
                let fan_out = self.layers[i].w.len();
                let xavier_stddev = (1.0 / ((fan_in + fan_out) as f32 / 2.0)).sqrt();
                let distribution = Normal::new(0.0, xavier_stddev).unwrap();
                let mut xavier = |elem: &mut T| {
                    *elem = T::from(distribution.sample(&mut rng));
                };

                self.layers[i].w.apply(&mut xavier);
                self.layers[i].b.apply(xavier);
            }
        }

        fn forward_matrix(&mut self, input: &DMatrix<T>) {
            //Layer 0 forwards the actual input
            self.layers[0].forward_matrix(input);
            //Following layers forard their previouse's layer .a matrix
            for i in 1..self.layers.len() {
                let inp = self.layers[i - 1].a.clone();
                self.layers[i].forward_matrix(&inp);
            }
        }

        fn forward_all(layers: &mut [FullyConnectedLayer<T>], input: &DMatrix<T>) {
            //Layer 0 forwards the actual input
            layers[0].forward_matrix(input);
            //Following layers forard their previouse's layer .a matrix
            for i in 1..layers.len() {
                let inp = layers[i - 1].a.clone();
                layers[i].forward_matrix(&inp);
            }
        }

        fn reset_deltas(&mut self) {
            for layer in self.layers.iter_mut() {
                layer.reset_deltas();
            }
        }

        pub fn train_stochastic(&mut self, lr: T, epoch: usize) -> T {
            let ncols = self.input_data.ncols();
            let num_samples = self.input_data.nrows() as f32;
            let ncols_out = self.output_data.ncols();
            let mut buffer = DMatrix::<T>::zeros(1, ncols);
            let mut target = DMatrix::<T>::zeros(1, ncols_out);
            let mut running_cost: T = T::zero();
            self.shuffle_network_data();
            for i in 0..self.input_data.nrows() {
                self.reset_deltas();
                for j in 0..ncols {
                    buffer[(0, j)] = self.input_data[(i, j)];
                }
                for j in 0..ncols_out {
                    target[(0, j)] = self.output_data[(i, j)];
                }
                self.forward_matrix(&buffer);
                let output_error = self.get_output_gradient(&target);
                running_cost += output_error.sum().powi(2) as T;
                self.backward(&buffer, &output_error, T::one());
                // self.optimize(lr);
                self.optimize_adamw(epoch + 1, lr);
            }
            running_cost / num_samples.into()
        }

        fn thread_train(
            thread_layers: &mut Vec<FullyConnectedLayer<T>>,
            offset: usize,
            input_data: &DMatrix<T>,
            output_data: &DMatrix<T>,
            batch_size: usize,
            thread_id: usize,
        ) -> T {
            let my_id = thread_id;
            let ncols = input_data.ncols();
            let ncols_out = output_data.ncols();
            let mut buffer = DMatrix::<T>::zeros(batch_size, ncols);
            let mut target = DMatrix::<T>::zeros(batch_size, ncols_out);
            let index = offset + batch_size * my_id;
            for k in 0..batch_size {
                for j in 0..ncols {
                    buffer[(k, j)] = input_data[(index + k, j)];
                }
                for j in 0..ncols_out {
                    target[(k, j)] = output_data[(index + k, j)];
                }
            }
            Network::<T>::forward_all(thread_layers, &buffer);
            let output_error = Network::<T>::get_output_gradient_all(thread_layers, &target);
            let running_cost = output_error.sum().powi(2) as T;
            Network::<T>::backward_all(thread_layers, &buffer, &output_error, T::one());
            Network::<T>::scale_all(thread_layers, batch_size);
            running_cost
        }

        pub fn train_minibatch_serial(&mut self, lr: T, _epoch: usize) -> T {
            let ncols = self.input_data.ncols();
            let num_samples = self.input_data.nrows() as f32;
            let ncols_out = self.output_data.ncols();
            let mut buffer = DMatrix::<T>::zeros(self.batch_size, ncols);
            let mut target = DMatrix::<T>::zeros(self.batch_size, ncols_out);
            let mut running_cost: T = T::zero();
            self.shuffle_network_data();
            self.zero_optimizer();
            for i in (0..self.input_data.nrows()).step_by(self.batch_size) {
                self.reset_deltas();
                if i + self.batch_size >= self.input_data.nrows() {
                    break;
                }
                for k in 0..self.batch_size {
                    for j in 0..ncols {
                        buffer[(k, j)] = self.input_data[(i + k, j)];
                    }
                    for j in 0..ncols_out {
                        target[(k, j)] = self.output_data[(i + k, j)];
                    }
                }
                Self::forward_all(&mut self.layers, &buffer);
                let output_error = Self::get_output_gradient_all(&self.layers, &target);
                running_cost += output_error.sum().powi(2) as T;
                Self::backward_all(&mut self.layers, &buffer, &output_error, T::one());
                Self::scale_all(&mut self.layers, self.batch_size);
                self.optimize_adamw(i + 1, lr);
            }
            running_cost / num_samples.into()
        }

        pub fn train_minibatch(&mut self, lr: T, epoch: usize, num_threads: usize) -> T {
            let batch_size = self.batch_size;
            let num_samples = self.input_data.nrows() as f32;
            let samples_per_loop = batch_size * num_threads;

            //If not threaded default to serial
            if num_threads <= 1 {
                return self.train_minibatch_serial(lr, epoch);
            }

            //Shuffle the input
            self.shuffle_network_data();
            self.zero_optimizer();

            let mut loops = 0;
            let mut running_cost: T = T::zero();
            loop {
                //Get the offset
                let offset = loops * samples_per_loop;

                //Break if we did go through all of our samples
                if offset + samples_per_loop >= self.input_data.nrows() {
                    break;
                }

                //Get a local copy of the network
                //TODO is this needed on every batch pass?
                self.m_thread_layers = vec![self.layers.clone(); num_threads];
                assert!(self.m_thread_layers.len() == num_threads);

                //Reset deltas for this round
                self.reset_deltas();

                // Reset thread layers to the main one
                self.m_thread_layers.par_iter_mut().for_each(|local_layer| {
                    *local_layer = self.layers.clone();
                });

                //Train each thread
                running_cost += self
                    .m_thread_layers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(id, local_layer)| -> T {
                        Self::thread_train(
                            local_layer,
                            offset,
                            &self.input_data,
                            &self.output_data,
                            self.batch_size,
                            id,
                        )
                    })
                    .collect::<Vec<T>>()
                    .iter()
                    .sum();

                for thread_layer in self.m_thread_layers.iter() {
                    for i in 0..self.layers.len() {
                        self.layers[i].dw += &thread_layer[i].dw;
                        self.layers[i].db += &thread_layer[i].db;
                    }
                }

                self.optimize_adamw(loops + 1, lr);
                loops += 1;
            }
            running_cost / num_samples.into()
        }

        fn scale(&mut self, n: usize) {
            let nn = n as f32;
            for l in self.layers.iter_mut() {
                l.dw /= nn.into();
                l.db /= nn.into();
            }
        }

        fn scale_all(layers: &mut [FullyConnectedLayer<T>], n: usize) {
            let nn = n as f32;
            for l in layers.iter_mut() {
                l.dw /= nn.into();
                l.db /= nn.into();
            }
        }

        fn optimize(&mut self, lr: T) {
            for l in self.layers.iter_mut() {
                l.w -= l.dw.scale(lr);
                l.b -= l.db.scale(lr);
            }
        }

        fn zero_optimizer(&mut self) {
            for l in self.layers.iter_mut() {
                l.m.fill(T::zero());
                l.v.fill(T::zero());
            }
        }

        fn optimize_adam(&mut self, iteration: usize, lr: T) {
            for l in self.layers.iter_mut() {
                l.m = l.m.scale(ADAM_BETA1.into()) + l.dw.scale(T::one() - ADAM_BETA1.into());
                l.v = l.v.scale(ADAM_BETA2.into())
                    + l.dw
                        .map(|x| T::powi(x, 2))
                        .scale(T::one() - ADAM_BETA2.into());
                let m_hat = &l.m / (T::one() - T::powi(ADAM_BETA1.into(), iteration as i32));
                let mut v_hat = &l.v / (T::one() - T::powi(ADAM_BETA2.into(), iteration as i32));
                v_hat.apply(|x| *x = T::sqrt(*x));
                v_hat.add_scalar_mut(ADAM_EPS.into());
                let update = (m_hat.component_div(&v_hat)).scale(lr);
                l.w -= update;
                l.b -= l.db.scale(lr);
            }
        }

        fn optimize_adamw(&mut self, iteration: usize, lr: T) {
            for l in self.layers.iter_mut() {
                l.m = l.m.scale(ADAM_BETA1.into()) + l.dw.scale(T::one() - ADAM_BETA1.into());
                l.v = l.v.scale(ADAM_BETA2.into())
                    + l.dw
                        .map(|x| T::powi(x, 2))
                        .scale(T::one() - ADAM_BETA2.into());
                let m_hat = &l.m / (T::one() - T::powi(ADAM_BETA1.into(), iteration as i32));
                let mut v_hat = &l.v / (T::one() - T::powi(ADAM_BETA2.into(), iteration as i32));
                v_hat.apply(|x| *x = T::sqrt(*x));
                v_hat.add_scalar_mut(ADAM_EPS.into());
                let decay = &l.w.scale(ADAM_DECAY.into());
                let update = &m_hat.component_div(&v_hat) + decay;
                l.w = &l.w - &update.scale(lr);
                l.b = &l.b - &l.db.scale(lr);
            }
        }

        fn backward(&mut self, input: &DMatrix<T>, output_error: &DMatrix<T>, scale: T) {
            let mut delta_next = output_error.clone();

            for i in (1..self.layers.len()).rev() {
                let prev_activation = &self.layers[i - 1].a.clone();
                let activation_prime =
                    FullyConnectedLayer::activate_prime_matrix(&self.layers[i].z);

                self.layers[i].delta = delta_next.component_mul(&activation_prime);
                let delta = &self.layers[i].delta.clone();

                self.layers[i].dw += prev_activation.transpose() * delta;
                self.layers[i].db += delta.row_sum();

                delta_next = &self.layers[i].delta * &self.layers[i].w.transpose();
            }

            //First layer at last!
            let first_layer = &mut self.layers[0];
            let activation_prime = FullyConnectedLayer::activate_prime_matrix(&first_layer.z);
            first_layer.delta = delta_next.component_mul(&activation_prime);
            first_layer.dw += input.transpose() * &first_layer.delta;
            first_layer.db += first_layer.delta.clone().row_sum();

            for l in self.layers.iter_mut() {
                l.dw /= scale;
                l.db /= scale;
            }
        }

        fn backward_all(
            layers: &mut [FullyConnectedLayer<T>],
            input: &DMatrix<T>,
            output_error: &DMatrix<T>,
            scale: T,
        ) {
            let mut delta_next = output_error.clone();

            for i in (1..layers.len()).rev() {
                let prev_activation = &layers[i - 1].a.clone();
                let activation_prime = FullyConnectedLayer::activate_prime_matrix(&layers[i].z);

                layers[i].delta = delta_next.component_mul(&activation_prime);
                let delta = &layers[i].delta.clone();

                layers[i].dw += prev_activation.transpose() * delta;
                layers[i].db += delta.row_sum();

                delta_next = &layers[i].delta * &layers[i].w.transpose();
            }

            //First layer at last!
            let first_layer = &mut layers[0];
            let activation_prime = FullyConnectedLayer::activate_prime_matrix(&first_layer.z);
            first_layer.delta = delta_next.component_mul(&activation_prime);
            first_layer.dw += input.transpose() * &first_layer.delta;
            first_layer.db += first_layer.delta.clone().row_sum();

            for l in layers.iter_mut() {
                l.dw /= scale;
                l.db /= scale;
            }
        }

        pub fn eval(&mut self, sample: &DMatrix<T>, output_buffer: &mut DMatrix<T>) {
            self.forward_matrix(sample);
            assert_eq!(
                output_buffer.ncols(),
                self.output_data.ncols(),
                "Dimension (1) mismatch in `eval()`."
            );
            assert_eq!(
                sample.ncols(),
                self.input_data.ncols(),
                "Dimension (2) mismatch in `eval()`."
            );
            let nn_output = &self.layers.last().unwrap().z;
            *output_buffer = nn_output.clone();
        }

        fn get_output_gradient(&mut self, target: &DMatrix<T>) -> DMatrix<T> {
            &self.layers.last().unwrap().a - target
        }

        fn get_output_gradient_all(
            layers: &[FullyConnectedLayer<T>],
            target: &DMatrix<T>,
        ) -> DMatrix<T> {
            &layers.last().unwrap().a - target
        }

        pub fn get_batchsize(&self) -> usize {
            self.batch_size
        }

        pub fn calculate_total_bytes(&self) -> usize {
            let mut total_size: usize = 0;
            for l in self.layers.iter() {
                total_size += l.total_bytes();
            }
            total_size
        }

        fn shuffle_network_data(&mut self) {
            let mut vec_in: Vec<Vec<T>> = self
                .input_data
                .row_iter()
                .map(|r| r.iter().copied().collect())
                .collect();

            let mut vec_out: Vec<Vec<T>> = self
                .output_data
                .row_iter()
                .map(|r| r.iter().copied().collect())
                .collect();

            let mut indices: Vec<usize> = (0..vec_in.len()).collect();
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);

            let mut shuffled_vec1: Vec<Vec<T>> = Vec::with_capacity(vec_in.len());
            let mut shuffled_vec2 = Vec::with_capacity(vec_in.len());

            for &i in &indices {
                shuffled_vec1.push(vec_in[i].clone());
                shuffled_vec2.push(vec_out[i].clone());
            }

            vec_in = shuffled_vec1;
            vec_out = shuffled_vec2;

            self.input_data =
                DMatrix::from_fn(vec_in.len() as _, vec_in[0].len() as _, |i, j| vec_in[i][j]);

            self.output_data =
                DMatrix::from_fn(vec_out.len() as _, vec_out[0].len() as _, |i, j| {
                    vec_out[i][j]
                });
        }
    }

    pub trait NetworkIO<T> {
        fn get_weights(&self) -> Vec<T>;
        fn quantize_mut(&mut self, bits: i32);
        fn load_weights(&mut self, weights: &[T]);
        fn get_network_state(&self) -> Vec<u8>;
        fn get_network_state_quantized(&self, bits: i32) -> Vec<u8>;
        fn new_from_file_with_batchsize(
            filename: &str,
            input_size: usize,
            output_size: usize,
            batch_size: usize,
        ) -> Result<Network<T>, Box<dyn std::error::Error>>;
        fn save(&self, filename: &str) {
            println!("Storing state to {}", filename);
            let bytes: Vec<u8> = self.get_network_state();
            let mut file = File::create(filename).expect("Failed to create file");
            file.write_all(&bytes)
                .expect("Failed to write state to file!");
        }
        fn save_quantized(&self, filename: &str, bits: i32) {
            println!("Storing state to {}", filename);
            let bytes: Vec<u8> = self.get_network_state_quantized(bits);
            let mut file = File::create(filename).expect("Failed to create file");
            file.write_all(&bytes)
                .expect("Failed to write quantized state to file!");
        }
    }

    fn vec_usize_to_bytes(vec: &Vec<usize>) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * std::mem::size_of::<usize>());
        for value in vec {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    impl NetworkIO<f32> for Network<f32> {
        fn get_network_state(&self) -> Vec<u8> {
            let arch_bytes = vec_usize_to_bytes(&self.neurons_per_layer);
            //Hardcoded 1 below is the 8 bytes we need to store the number of layers
            let total_bytes = self.calculate_total_bytes()
                + std::mem::size_of::<usize>() * (1 + 1)
                + arch_bytes.len()
                + 12;
            let mut data: Vec<u8> = vec![];
            data.reserve_exact(total_bytes);
            let num_layers = self.layers.len();
            let bytes: [u8; std::mem::size_of::<usize>()] = num_layers.to_ne_bytes();
            data.extend_from_slice(&bytes);
            data.extend_from_slice(&arch_bytes.len().to_ne_bytes());
            data.extend_from_slice(&arch_bytes);
            let weights = self.get_weights();
            let weights_in_bytes = vec_to_bytes(&weights);
            let bits: i32 = 0;
            let delta: f32 = 0.0;
            let min_val: f32 = 0.0;
            data.extend_from_slice(&bits.to_ne_bytes());
            data.extend_from_slice(&delta.to_ne_bytes());
            data.extend_from_slice(&min_val.to_ne_bytes());
            data.extend_from_slice(&weights_in_bytes);
            println!("Bytes serialized {}.", data.len());
            data
        }
        fn get_network_state_quantized(&self, bits: i32) -> Vec<u8> {
            let arch_bytes = vec_usize_to_bytes(&self.neurons_per_layer);
            //Hardcoded 1 below is the 8 bytes we need to store the number of layers
            let total_bytes = self.calculate_total_bytes()
                + std::mem::size_of::<usize>() * (1 + 1)
                + arch_bytes.len()
                + 12;
            let mut data: Vec<u8> = vec![];
            data.reserve_exact(total_bytes);
            let num_layers = self.layers.len();
            let bytes: [u8; std::mem::size_of::<usize>()] = num_layers.to_ne_bytes();
            data.extend_from_slice(&bytes);
            data.extend_from_slice(&arch_bytes.len().to_ne_bytes());
            data.extend_from_slice(&arch_bytes);
            let weights = self.get_weights();
            let (delta, min_val, quantized_weights) = quantize_vector(&weights, bits);
            let packed_quantized_weights: Vec<u8> =
                pack_quantization_indices(&quantized_weights, bits);
            // let weights_in_bytes = vec_to_bytes(quantized_weights);
            let weights_in_bytes = packed_quantized_weights;
            data.extend_from_slice(&bits.to_ne_bytes());
            data.extend_from_slice(&delta.to_ne_bytes());
            data.extend_from_slice(&min_val.to_ne_bytes());
            data.extend_from_slice(&weights_in_bytes);
            println!("Bytes serialized {}.", data.len());
            data
        }

        fn new_from_file_with_batchsize(
            filename: &str,
            input_size: usize,
            output_size: usize,
            batch_size: usize,
        ) -> Result<Network<f32>, Box<dyn std::error::Error>> {
            let mut file = File::open(filename)?;
            let mut buffer: Vec<u8> = Vec::new();

            let _bytes_read = file.read_to_end(&mut buffer)?;
            let mut offset = 0;
            let _num_layers =
                usize::from_ne_bytes(buffer[offset..(offset + 8)].try_into().unwrap());
            offset += 8;
            let arch_size = usize::from_ne_bytes(buffer[offset..(offset + 8)].try_into().unwrap());
            offset += 8;
            let mut hidden_layers = Vec::<usize>::new();
            for _ in (0..arch_size).step_by(std::mem::size_of::<usize>()) {
                hidden_layers.push(usize::from_ne_bytes(
                    buffer[offset..(offset + 8)].try_into().unwrap(),
                ));
                offset += 8;
            }
            let fake_input = DMatrix::<f32>::zeros(1, input_size);
            let fake_output = DMatrix::<f32>::zeros(1, output_size);
            let mut net = Network::<f32>::new(
                input_size,
                output_size,
                hidden_layers,
                &fake_input,
                &fake_output,
                batch_size,
            );
            // Read in Quantiazation bits delta and min val;
            let q = i32::from_ne_bytes(buffer[offset..(offset + 4)].try_into().unwrap());
            offset += 4;
            let delta = f32::from_ne_bytes(buffer[offset..(offset + 4)].try_into().unwrap());
            offset += 4;
            let min_val = f32::from_ne_bytes(buffer[offset..(offset + 4)].try_into().unwrap());
            offset += 4;
            let weights: Vec<f32> =
                unpack_quantization_indices(&mut buffer[offset..], delta, min_val, q);
            net.load_weights(&weights);
            Ok(net)
        }

        fn get_weights(&self) -> Vec<f32> {
            let mut weights: Vec<f32> = Vec::new();
            for l in self.layers.iter() {
                for val in l.w.iter() {
                    weights.push(*val);
                }
                for val in l.b.iter() {
                    weights.push(*val);
                }
            }
            weights
        }

        fn load_weights(&mut self, weights: &[f32]) {
            let mut offset = 0;
            for l in self.layers.iter_mut() {
                for val in l.w.iter_mut() {
                    *val = weights[offset];
                    offset += 1;
                }
                for val in l.b.iter_mut() {
                    *val = weights[offset];
                    offset += 1;
                }
            }
        }

        fn quantize_mut(&mut self, bits: i32) {
            let weights = self.get_weights();
            let (delta, min_val, quants) = quantize_vector(&weights, bits);
            let w = dequantize_vector(delta, min_val, &quants);
            self.load_weights(&w);
        }
    }

    impl NetworkIO<f64> for Network<f64> {
        fn get_network_state(&self) -> Vec<u8> {
            let arch_bytes = vec_usize_to_bytes(&self.neurons_per_layer);
            //Hardcoded 1 below is the 8 bytes we need to store the number of layers
            let total_bytes = self.calculate_total_bytes()
                + std::mem::size_of::<usize>() * (1 + 1)
                + arch_bytes.len()
                + 20;
            let mut data: Vec<u8> = vec![];
            data.reserve_exact(total_bytes);
            let num_layers = self.layers.len();
            let bytes: [u8; std::mem::size_of::<usize>()] = num_layers.to_ne_bytes();
            data.extend_from_slice(&bytes);
            data.extend_from_slice(&arch_bytes.len().to_ne_bytes());
            data.extend_from_slice(&arch_bytes);
            let weights = self.get_weights();
            let weights_in_bytes = vec_to_bytes(&weights);
            let bits: i32 = 0;
            let delta: f64 = 0.0;
            let min_val: f64 = 0.0;
            data.extend_from_slice(&bits.to_ne_bytes());
            data.extend_from_slice(&delta.to_ne_bytes());
            data.extend_from_slice(&min_val.to_ne_bytes());
            data.extend_from_slice(&weights_in_bytes);
            println!("Bytes serialized {}.", data.len());
            data
        }

        fn get_network_state_quantized(&self, bits: i32) -> Vec<u8> {
            let arch_bytes = vec_usize_to_bytes(&self.neurons_per_layer);
            //Hardcoded 1 below is the 8 bytes we need to store the number of layers
            let total_bytes = self.calculate_total_bytes()
                + std::mem::size_of::<usize>() * (1 + 1)
                + arch_bytes.len()
                + 20;
            let mut data: Vec<u8> = vec![];
            data.reserve_exact(total_bytes);
            let num_layers = self.layers.len();
            let bytes: [u8; std::mem::size_of::<usize>()] = num_layers.to_ne_bytes();
            data.extend_from_slice(&bytes);
            data.extend_from_slice(&arch_bytes.len().to_ne_bytes());
            data.extend_from_slice(&arch_bytes);
            let weights = self.get_weights();
            let (delta, min_val, quantized_weights) = quantize_vector(&weights, bits);
            let packed_quantized_weights: Vec<u8> =
                pack_quantization_indices(&quantized_weights, bits);
            let weights_in_bytes = packed_quantized_weights;
            data.extend_from_slice(&bits.to_ne_bytes());
            data.extend_from_slice(&delta.to_ne_bytes());
            data.extend_from_slice(&min_val.to_ne_bytes());
            data.extend_from_slice(&weights_in_bytes);
            println!("Bytes serialized {}.", data.len());
            data
        }

        fn new_from_file_with_batchsize(
            filename: &str,
            input_size: usize,
            output_size: usize,
            batch_size: usize,
        ) -> Result<Network<f64>, Box<dyn std::error::Error>> {
            let mut file = File::open(filename)?;
            let mut buffer: Vec<u8> = Vec::new();

            let _bytes_read = file.read_to_end(&mut buffer)?;
            let mut offset = 0;
            let _num_layers =
                usize::from_ne_bytes(buffer[offset..(offset + 8)].try_into().unwrap());
            offset += 8;
            let arch_size = usize::from_ne_bytes(buffer[offset..(offset + 8)].try_into().unwrap());
            offset += 8;
            let mut hidden_layers = Vec::<usize>::new();
            for _ in (0..arch_size).step_by(std::mem::size_of::<usize>()) {
                hidden_layers.push(usize::from_ne_bytes(
                    buffer[offset..(offset + 8)].try_into().unwrap(),
                ));
                offset += 8;
            }
            let fake_input = DMatrix::<f64>::zeros(1, input_size);
            let fake_output = DMatrix::<f64>::zeros(1, output_size);
            let mut net = Network::<f64>::new(
                input_size,
                output_size,
                hidden_layers,
                &fake_input,
                &fake_output,
                batch_size,
            );
            //Read in Quantiazation bits delta and min val;
            let q = i32::from_ne_bytes(buffer[offset..(offset + 4)].try_into().unwrap());
            offset += 4;
            let delta = f64::from_ne_bytes(buffer[offset..(offset + 8)].try_into().unwrap());
            offset += 8;
            let min_val = f64::from_ne_bytes(buffer[offset..(offset + 8)].try_into().unwrap());
            offset += 8;
            let weights: Vec<f64> =
                unpack_quantization_indices(&mut buffer[offset..], delta, min_val, q);
            net.load_weights(&weights);
            Ok(net)
        }

        fn get_weights(&self) -> Vec<f64> {
            let mut weights: Vec<f64> = Vec::new();
            for l in self.layers.iter() {
                for val in l.w.iter() {
                    weights.push(*val);
                }
                for val in l.b.iter() {
                    weights.push(*val);
                }
            }
            weights
        }

        fn load_weights(&mut self, weights: &[f64]) {
            let mut offset = 0;
            for l in self.layers.iter_mut() {
                for val in l.w.iter_mut() {
                    *val = weights[offset];
                    offset += 1;
                }
                for val in l.b.iter_mut() {
                    *val = weights[offset];
                    offset += 1;
                }
            }
        }

        fn quantize_mut(&mut self, bits: i32) {
            let weights = self.get_weights();
            let (delta, min_val, quants) = quantize_vector(&weights, bits);
            let w = dequantize_vector(delta, min_val, &quants);
            self.load_weights(&w);
        }
    }
}
