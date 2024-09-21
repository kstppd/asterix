mod network;
use crate::network::network::Network;
use nalgebra::DMatrix;
use rand::distributions::Uniform;
use rand::Rng;

fn vdf_fourier_features(
    vdf: &[f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    order: usize,
    size: usize,
) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let total_dims = 3 + order * 6;

    let mut harmonics: Vec<f64> = vec![0.0; order];
    harmonics.resize(order, 0.0);
    let mut rng = rand::thread_rng();
    let range = Uniform::<f64>::new(-Into::<f64>::into(0.0), Into::<f64>::into(6.0));
    harmonics.iter_mut().for_each(|x| *x = rng.sample(range));

    let mut vspace = DMatrix::<f64>::zeros(vdf.len(), total_dims);
    let mut density = DMatrix::<f64>::zeros(vdf.len(), 1);

    // Iterate over pixels
    for counter in 0..size {
        let pos_z = vz[counter] - 0.5_f64;
        let pos_y = vy[counter] - 0.5_f64;
        let pos_x = vx[counter] - 0.5_f64;
        assert!((-0.5..=0.5).contains(&pos_z));
        assert!((-0.5..=0.5).contains(&pos_x));
        assert!(pos_z <= 0.5 && pos_y >= -0.5);
        vspace[(counter, 0)] = pos_x;
        vspace[(counter, 1)] = pos_y;
        vspace[(counter, 2)] = pos_z;
        for f in 0..order {
            vspace[(counter, f * 6 + 3)] =
                (harmonics[f] * 2.0 * std::f64::consts::PI * pos_x).sin();
            vspace[(counter, f * 6 + 4)] =
                (harmonics[f] * 2.0 * std::f64::consts::PI * pos_y).sin();
            vspace[(counter, f * 6 + 5)] =
                (harmonics[f] * 2.0 * std::f64::consts::PI * pos_z).sin();
            vspace[(counter, f * 6 + 6)] =
                (harmonics[f] * 2.0 * std::f64::consts::PI * pos_x).cos();
            vspace[(counter, f * 6 + 7)] =
                (harmonics[f] * 2.0 * std::f64::consts::PI * pos_y).cos();
            vspace[(counter, f * 6 + 8)] =
                (harmonics[f] * 2.0 * std::f64::consts::PI * pos_z).cos();
        }
        density[(counter, 0)] = vdf[counter];
    }
    (vspace, density, harmonics)
}

fn scale_vdf(vdf: &mut [f64], sparse: f64) {
    vdf.iter_mut()
        .for_each(|x| *x = f64::abs(f64::log10(f64::max(*x, 0.001 * sparse))));
}

fn unscale_vdf(vdf: &mut [f64]) {
    vdf.iter_mut().for_each(|x| {
        *x = f64::powf(10.0, -1.0 * *x);
    });
}

fn sparsify(vdf: &mut [f64], sparse: f64) {
    vdf.iter_mut().for_each(|x| {
        if *x - sparse <= 0.0 {
            *x = 0.0;
        }
    });
}

fn normalize_vdf(vdf: &mut [f64]) -> (f64, f64) {
    let min_val = *vdf.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_val = *vdf.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let range = max_val - min_val;
    vdf.iter_mut().for_each(|x| *x = (*x - min_val) / range);
    (min_val, max_val)
}

fn unnormalize_vdf(vdf: &mut [f64], min_val: f64, max_val: f64) {
    let range = max_val - min_val;
    vdf.iter_mut().for_each(|x| *x = *x * range + min_val);
}

fn reconstruct_vdf(net: &mut Network<f64>, vspace: &DMatrix<f64>) -> Vec<f64> {
    let mut sample = DMatrix::<f64>::zeros(1, vspace.ncols());
    let mut buffer = DMatrix::<f64>::zeros(1, 1);
    let mut reconstructed_vdf: Vec<f64> = vec![];
    let mut decoder = Network::<f64>::new_from_other_with_batchsize(net, 1);
    for s in 0..vspace.nrows() {
        for i in 0..vspace.ncols() {
            sample[(0, i)] = vspace[(s, i)];
        }
        decoder.eval(&sample, &mut buffer);
        reconstructed_vdf.push(buffer[(0, 0)]);
    }
    reconstructed_vdf
}

fn compress_vdf(
    vdf: &[f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    fourier_order: usize,
    epochs: usize,
    hidden_layers: Vec<usize>,
    size: usize,
    tol: f64,
) -> (Vec<f64>, usize) {
    let (vspace, density, _harmonics) = vdf_fourier_features(vdf, vx, vy, vz, fourier_order, size);
    let mut net = Network::<f64>::new(
        vspace.ncols(),
        density.ncols(),
        hidden_layers,
        &vspace,
        &density,
        8,
    );
    net.randomize_he();
    //Train
    let mut epoch = 0;
    loop {
        let cost = net.train_minibatch(2.5e-5, epoch, 1);
        // if epoch % 1 == 0 {
        //     println!("Cost at epoch {} is {:.7}", epoch, cost);
        // }
        if cost < tol || epoch > epochs {
            // println!("Breaking  at epoch {} and cost is {:.6}", epoch, cost);
            break;
        }
        epoch += 1;
    }

    let reconstructed = reconstruct_vdf(&mut net, &vspace);
    let bytes_used = net.calculate_total_bytes();
    (reconstructed, bytes_used)
}

#[no_mangle]
pub extern "C" fn compress_and_reconstruct_vdf(
    vx_ptr: *const f64,
    vy_ptr: *const f64,
    vz_ptr: *const f64,
    vspace_ptr: *const f32,
    size: usize,
    new_vspace_ptr: *mut f32,
    max_epochs: usize,
    fourier_order: usize,
    hidden_layers_ptr: *const usize,
    n_hidden_layers: usize,
    sparse: f64,
    tol: f64,
) -> f64 {
    let vdf_f32 = unsafe { std::slice::from_raw_parts(vspace_ptr, size).to_vec() };
    let mut vdf: Vec<f64> = vdf_f32.iter().map(|&x| x as f64).collect();
    let vx = unsafe { std::slice::from_raw_parts(vx_ptr, size) };
    let vy = unsafe { std::slice::from_raw_parts(vy_ptr, size) };
    let vz = unsafe { std::slice::from_raw_parts(vz_ptr, size) };
    let hidden_layers =
        unsafe { std::slice::from_raw_parts(hidden_layers_ptr, n_hidden_layers).to_vec() };
    scale_vdf(&mut vdf, sparse);
    let norm = normalize_vdf(&mut vdf);
    let (mut reconstructed, bytes_used) = compress_vdf(
        &vdf,
        vx,
        vy,
        vz,
        fourier_order,
        max_epochs,
        hidden_layers,
        size,
        tol,
    );
    unnormalize_vdf(&mut reconstructed, norm.0, norm.1);
    unscale_vdf(&mut reconstructed);
    sparsify(&mut reconstructed, sparse);
    for (i, v) in reconstructed.iter().enumerate() {
        unsafe { *new_vspace_ptr.add(i) = *v as f32 };
    }
    size as f64 * std::mem::size_of::<f64>() as f64 / bytes_used as f64
}
