
module HEXA

using LinearAlgebra, SparseArrays, Printf, Base.Threads, JSON, Dates
using CUDA
using YAML

include("Helpers.jl")
include("Element.jl")
include("Mesh.jl")
include("Boundary.jl")
include("ExportVTK.jl")
include("Stress.jl")


include("SOLVER/CPUSolver.jl")
include("SOLVER/GPUSolver.jl")
include("SOLVER/IterativeSolver.jl")
include("SOLVER/DirectSolver.jl")
include("SOLVER/Solver.jl")


include("Configuration.jl")
include("TopologyOptimization.jl")
include("Postprocessing.jl")


using .Helpers
using .Element
using .Mesh
using .Boundary
using .ExportVTK
using .Stress
using .CPUSolver
using .GPUSolver
using .Solver
using .Configuration
using .TopologyOptimization
using .Postprocessing

function __init__()
    println("HEXA Finite Element Solver initialized")
    println("Active Threads = $(Threads.nthreads())")
    println("Clearing GPU memory...")
    Helpers.clear_gpu_memory()
end

"""
    run_main(config_file=nothing)

Run the main HEXA simulation using the configuration from the specified JSON/YAML file.
"""
function run_main(config_file=nothing)
    
    if config_file === nothing
        config_file = "config.yaml"
    end
    

    println("Loading configuration from: $config_file")
    config = load_configuration(config_file)
    
    geom = setup_geometry(config)
    
    
    
    
    
    config["geometry"]["nElem_x_computed"] = geom.nElem_x
    config["geometry"]["nElem_y_computed"] = geom.nElem_y
    config["geometry"]["nElem_z_computed"] = geom.nElem_z
    config["geometry"]["dx_computed"] = geom.dx
    config["geometry"]["dy_computed"] = geom.dy
    config["geometry"]["dz_computed"] = geom.dz
    
    config["geometry"]["max_domain_dim"] = geom.max_domain_dim
    

    
    println("\nGenerating structured prismatic mesh...")
    nodes, elements, dims = generate_mesh(
        geom.nElem_x, geom.nElem_y, geom.nElem_z;
        dx = geom.dx, dy = geom.dy, dz = geom.dz
    )
    
    nNodes = size(nodes, 1)
    nElem = size(elements, 1)
    println("Final mesh statistics:")
    println("  Nodes: $(nNodes)")
    println("  Elements: $(nElem)")
    
    bc_data = config["boundary_conditions"]
    
    bc_indicator = get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
    
    
    E = Float32(config["material"]["E"])
    nu = Float32(config["material"]["nu"])
    println("Material properties: E=$(E), nu=$(nu)")
    
    ndof = nNodes * 3
    F = zeros(Float32, ndof)
    forces_data = config["external_forces"]
    
    apply_external_forces!(F, Vector{Any}(forces_data), nodes, elements)
    
    
    number_of_iterations = get(config, "number_of_iterations", 0)
    l1_stress_allowable = Float32(get(config, "l1_stress_allowable", 1.0))

    if l1_stress_allowable == 0.0f0
        @warn "l1_stress_allowable is 0. Setting to 1.0f0 to avoid division by zero."
        l1_stress_allowable = 1.0f0
    end
    
    
    density, original_density, protected_elements_mask = 
        initialize_density_field(nodes, elements, geom.shapes_to_add, geom.shapes_to_remove, config)
    

    
    opt_params = config["optimization_parameters"]
    min_density = Float32(get(opt_params, "min_density", 1.0e-3))
    max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))

    
    base_name = splitext(basename(config_file))[1]

    
    U_full = zeros(Float32, ndof)
    principal_field = zeros(Float32, 3, nElem)
    vonmises_field = zeros(Float32, nElem)
    full_stress_voigt = zeros(Float32, 6, nElem)
    l1_stress_norm_field = zeros(Float32, nElem)

    
    # --- ITERATION LOOP CONTROL ---
    
    RESULTS_DIR = "RESULTS"
    mkpath(RESULTS_DIR)
    println("All outputs will be saved to: $RESULTS_DIR")

    if number_of_iterations == 0
        println("number_of_iterations = 0, performing a single analysis.")
    else
        println("Starting iterative analysis. Base iterations: $number_of_iterations")
    end

    iter = 1
    keep_running = true
    is_annealing = false
    max_annealing_iters = 100 # Safety brake for infinite loops
    convergence_threshold = 0.01 # 1% change

    while keep_running
        
        # Determine Phase
        if number_of_iterations > 0 && iter > number_of_iterations
            is_annealing = true
            annealing_idx = iter - number_of_iterations
            
            println("\n--- Starting Annealing Iteration $annealing_idx (Global: $iter) ---")
            
            # Safety break
            if annealing_idx > max_annealing_iters
                println("⚠️ Max annealing iterations ($max_annealing_iters) reached. Stopping.")
                break
            end
        elseif number_of_iterations > 0
            println("\n--- Starting Iteration $(iter) / $(number_of_iterations) ---")
        end

        # Ensure "soft" and "hard" elements are reset
        if iter > 1
            println("Verifying protected element densities before FE analysis...")
            for e in 1:nElem
                if protected_elements_mask[e]
                    density[e] = original_density[e]
                end
            end
        end

        
        println("Calling unified solver...")
        U_full = Solver.solve_system(
            nodes, elements, E, nu, bc_indicator, F;
            density=density,
            config=config,
            
            min_stiffness_threshold=min_density
            
        )
        
        
        principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field =
            compute_stress_field(nodes, elements, U_full, E, nu, density)
        

        
        
        export_iteration_results(
            iter, base_name, RESULTS_DIR, nodes, elements,
            U_full, F, bc_indicator, principal_field,
            vonmises_field, full_stress_voigt,
            l1_stress_norm_field, density, E,
            geom 
        )
        

        
        # --- DENSITY UPDATE & CONVERGENCE CHECK ---
        
        if number_of_iterations > 0
            
            # Update density and get change metric
            # Pass 'is_annealing' to control threshold (98%) and radius (>= 1 elem)
            max_change = update_density!(
                density, l1_stress_norm_field, protected_elements_mask,
                E, l1_stress_allowable, iter, number_of_iterations,
                original_density,
                min_density, max_density_clamp,
                config,
                is_annealing
            )
            
            # Convergence Logic
            if is_annealing
                if max_change < convergence_threshold
                    println("\n✅ CONVERGENCE REACHED: Topology change $(@sprintf("%.4f%%", max_change*100)) < 1.0%")
                    keep_running = false
                else
                    println("⏳ Annealing... Change: $(@sprintf("%.4f%%", max_change*100)) (Target: < 1.0%)")
                end
            elseif iter == number_of_iterations
                println("Base iterations complete. Entering annealing phase...")
            end
            
        else
            # Single run (iter=1, num_iters=0), stop immediately
            keep_running = false
        end

        
        if CUDA.functional()
            println("Clearing GPU memory between iterations...")
            Helpers.clear_gpu_memory()
        end
        
        iter += 1
    end

    println("Clearing GPU memory...")
    Helpers.clear_gpu_memory()
    
    return (
        nodes = nodes,
        elements = elements,
        displacements = U_full,
        principal_stress = principal_field,
        vonmises_stress = vonmises_field,
        l1_stress_norm = l1_stress_norm_field,
        stress_tensor = full_stress_voigt
    )
end

end

using .HEXA

config_path = joinpath(@__DIR__, "config.yaml")

user_prompt_config = raw"H:\WORK\PhD\HEXAS\2025_11_26_HEXAS_TOPO\HEXAS_FEM-main\config.yaml"


if isfile(user_prompt_config)
    HEXA.run_main(user_prompt_config)
elseif isfile(config_path)
    println("User-specified config not found, using default: $config_path")
    HEXA.run_main(config_path)
else
    @error "Neither user-specified config nor default config.yaml found."
end