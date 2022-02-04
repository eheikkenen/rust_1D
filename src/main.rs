#![allow(dead_code)]
#![feature(array_windows)]

// a rust implementation of a simple 1D explicit finite element diffusion simulation with insulated boundary conditions

use rayon::prelude::*;
use std::fs;
use std::time::Instant;

fn main() {
    // initial variables
    let mesh_unit_length = 8;
    let start_time = 0.0;
    let end_time = 6.0;
    let time_step_factor = 0.95;
    let output_frequency = 2.0;
    let diffusivity = 0.1;
    let mesh_start = 0.0;
    let mesh_end = 6.0;

    // calculate other variables
    let dx = 1.0 / mesh_unit_length as f64;
    let mesh_length = (mesh_end as i32 - mesh_start as i32) * mesh_unit_length;
    let dt_explicit = dx * dx / (2.0 * diffusivity); // CFL condition
    let time_step = time_step_factor * dt_explicit; // ensures stability

    // mesh
    // look into scan method .iter().scan(initial, closure)
    let mut x: Vec<f64> = vec![0.0; (mesh_length + 2) as usize];
    x[0] = mesh_start;
    x[1] = mesh_start + dx / 2.0;
    x[(mesh_length + 1) as usize] = mesh_end;
    for i in 2..(mesh_length + 1) as usize {
        x[i] = x[1] + (i as f64 - 1.0) * dx;
    }

    // initialize
    let mut time = start_time;
    let mut output_time = output_frequency;
    let final_time_step = (end_time / time_step) + 1.0;

    // set initial conditions
    let mut density: Vec<f64> = x
        .iter()
        .take((mesh_length + 1) as usize)
        .map(|&val| if val >= 1.0 && val <= 2.0 { 5.0 } else { 0.0 })
        .collect();
    density.push(0.0);

    println!("t={:?}", time);
    println!("x={:?}", x);
    println!("u={:?}", density);
    println!("area={:?}", trapezoidal_rule(&density, mesh_length, dx));

    let initial_area = trapezoidal_rule(&density, mesh_length, dx);

    // time stepping
    let now = Instant::now();
    for i in 1..final_time_step as i32 {
        // get flux
        let flux = get_flux(&x, &density, diffusivity, mesh_length);

        // calc pde
        density
            .iter_mut()
            // .par_iter_mut()
            .skip(1)
            .take(mesh_length as usize)
            .zip(flux.windows(2))
            // .zip(flux.par_windows(2))
            .for_each(|(density, flux)| *density += time_step / dx * (flux[0] - flux[1]));
        density[0 as usize] = density[1];
        density[(mesh_length + 1) as usize] = density[mesh_length as usize];

        //iterate time
        time = i as f64 * time_step;

        //output
        if time >= output_time {
            println!("t={:?}", time);
            // println!("f={:.2?}", flux);
            println!("u={:.2?}", density);
            println!("area={:?}", trapezoidal_rule(&density, mesh_length, dx));
            output_time += output_frequency;
        }
    }

    // final output and timing
    println!("t={:?}", time);
    println!("u={:.2?}", density);
    println!("area={:?}", trapezoidal_rule(&density, mesh_length, dx));
    let elapsed_time = now.elapsed();
    println!("\nElapsed: {:.2?}", elapsed_time);
    let final_area = trapezoidal_rule(&density, mesh_length, dx);
    println!("Final Error: {:?}", (final_area - initial_area).abs());
}

fn get_flux(x: &Vec<f64>, density: &Vec<f64>, diffusivity: f64, mesh_length: i32) -> Vec<f64> {
    let mut flux: Vec<f64> = density
        .array_windows::<2>()
        .zip(x.array_windows::<2>())
        .skip(1)
        .take((mesh_length - 1) as usize)
        .map(|([d0, d1], [x0, x1])| -diffusivity * (d1 - d0) / (x1 - x0))
        .collect();
    // set insulated boundary conditions
    flux.insert(0, 0.0);
    flux.push(0.0);
    flux
}

// calculates trapezoidal rule to ensure conservation of area
fn trapezoidal_rule(u: &Vec<f64>, m: i32, dx: f64) -> f64 {
    let mut area: f64 = (u[0] - u[1] - u[m as usize] + u[(m + 1) as usize]) / 4.0;
    for i in 1..=m as usize {
        area += u[i];
    }
    area * dx
}
