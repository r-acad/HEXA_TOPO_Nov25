module Solver 
 
using CUDA 
using ..Helpers  
using ..DirectSolver: solve_system as solve_system_direct 
using ..IterativeSolver: solve_system_iterative 
 
 
export solve_system 
 
""" 
    choose_solver(nNodes, nElem, config) 
 
Determine the appropriate solver type based on config, problem size and available hardware. 
""" 
function choose_solver(nNodes, nElem, config) 
    solver_params = config["solver_parameters"] 
    configured_type = Symbol(lowercase(get(solver_params, "solver_type", "direct"))) 
 
    if configured_type == :direct 
        if nElem > 100_000 
            @warn "Direct solver requested for large mesh ($(nElem) elements). Switching to Matrix-Free iterative." 
            return :matrix_free 
        end 
        return :direct 
    elseif configured_type == :gpu 
        if CUDA.functional() && Helpers.has_enough_gpu_memory(nNodes, nElem) 
            println("GPU is available with sufficient memory. Using GPU solver.") 
            return :gpu 
        else 
            if CUDA.functional() 
                println("GPU is available but insufficient memory; falling back to CPU solver.") 
            else 
                println("No GPU available; falling back to CPU solver.") 
            end 
            return :matrix_free 
        end 
    elseif configured_type == :matrix_free 
        return :matrix_free 
    else 
        @warn "Unknown solver_type: $(configured_type). Defaulting to matrix_free." 
        return :matrix_free 
    end 
end 
 
""" 
    solve_system(...) 
 
Single entry point for all linear solvers. 
Dispatches to the correct solver based on problem size and density. 
""" 
function solve_system(nodes::Matrix{Float32}, 
                      elements::Matrix{Int}, 
                      E::Float32, 
                      nu::Float32, 
                      bc_indicator::Matrix{Float32}, 
                      F::Vector{Float32}; 
                      density::Vector{Float32}=nothing, 
                      config::Dict, 
                       
                      min_stiffness_threshold::Float32=Float32(1.0e-3)) 
                       
     
    nNodes = size(nodes, 1) 
    nElem = size(elements, 1) 
     
    solver_params = config["solver_parameters"] 
     
    solver_type = choose_solver(nNodes, nElem, config) 
     
     
    tol = Float32(get(solver_params, "tolerance", 1.0e-6)) 
    max_iter = Int(get(solver_params, "max_iterations", 1000)) 
    shift_factor = Float32(get(solver_params, "diagonal_shift_factor", 1.0e-6)) 
     
     
    use_precond = true 
    if solver_type == :gpu 
         
         
         
         
        # @warn "GPU solver forced to use UNPRECONDITIONED mode to bypass CUSPARSE IC(0) console dump/failure." 
    end 
     
 
 
    println("Dispatching to solver: :$(solver_type) with TOL=$(tol), MAX_ITER=$(max_iter)") 
 
    U_full = if solver_type == :direct 
        solve_system_direct(nodes, elements, E, nu, bc_indicator, F; 
                            density=density, 
                            shift_factor=shift_factor, 
                             
                            min_stiffness_threshold=min_stiffness_threshold) 
                             
    elseif solver_type == :gpu 
        gpu_method = Symbol(lowercase(get(solver_params, "gpu_method", "krylov"))) 
        krylov_solver = Symbol(lowercase(get(solver_params, "krylov_solver", "cg"))) 
 
        solve_system_iterative(nodes, elements, E, nu, bc_indicator, F; 
                             solver_type=:gpu, max_iter=max_iter, tol=tol, 
                             density=density, 
                             use_precond=use_precond,  
                             gpu_method=gpu_method, krylov_solver=krylov_solver, 
                             shift_factor=shift_factor, 
                              
                             min_stiffness_threshold=min_stiffness_threshold) 
                              
    else  
         
        solve_system_iterative(nodes, elements, E, nu, bc_indicator, F; 
                             solver_type=:matrix_free, max_iter=max_iter, tol=tol, 
                             use_precond=true, 
                             density=density, 
                             shift_factor=shift_factor, 
                              
                             min_stiffness_threshold=min_stiffness_threshold) 
                              
    end 
 
    return U_full 
end 
 
end 
 
 