module GPUSolver

using LinearAlgebra, SparseArrays, Printf
using CUDA          
using CUDA.CUSPARSE     
using Krylov, LinearOperators  
using ..Element

export solve_system_gpu

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function print_section_header(title::String)
    width = 80
    println("\n" * "="^width)
    padding = (width - length(title) - 2) ÷ 2
    println(" "^padding * title)
    println("="^width)
end

function print_subsection(title::String)
    println("\n" * "-"^80)
    println("  $title")
    println("-"^80)
end

function format_memory(bytes::Float64)
    if bytes >= 1024^3
        return @sprintf("%.2f GB", bytes / 1024^3)
    elseif bytes >= 1024^2
        return @sprintf("%.2f MB", bytes / 1024^2)
    else
        return @sprintf("%.2f KB", bytes / 1024)
    end
end

function get_free_dofs(bc_indicator::Matrix{T}) where T
    nNodes = size(bc_indicator, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    @inbounds for i in 1:nNodes
        if bc_indicator[i,1] > 0; constrained[3*(i-1)+1] = true; end
        if bc_indicator[i,2] > 0; constrained[3*(i-1)+2] = true; end
        if bc_indicator[i,3] > 0; constrained[3*(i-1)+3] = true; end
    end
    return findall(!, constrained)
end

# ============================================================================
# ASSEMBLY - OPTIMIZED WITH BETTER LOGGING
# ============================================================================

function assemble_sparse_matrix_optimized(nodes::Matrix{T}, elements::Matrix{Int}, 
                                          E::T, nu::T, density::Vector{T}, 
                                          min_stiffness_threshold::T) where T
    
    print_section_header("GPU SOLVER - MATRIX ASSEMBLY")
    
    assembly_start = time()
    
    nElem = size(elements, 1)
    nNodes = size(nodes, 1)
    ndof = nNodes * 3

    # Filter active elements
    active_indices = findall(d -> d >= min_stiffness_threshold, density)
    nActive = length(active_indices)
    
    if nActive == 0
        error("❌ No active elements found. Check density initialization or threshold.")
    end

    @printf("  Total elements:        %12d\n", nElem)
    @printf("  Active elements:       %12d (%.1f%%)\n", nActive, 100.0 * nActive / nElem)
    @printf("  Total DOFs:            %12d\n", ndof)
    @printf("  Density threshold:     %12.6f\n", min_stiffness_threshold)
    println()

    # Compute canonical stiffness matrix
    canonical_start = time()
    n1 = nodes[elements[1,1], :]
    n2 = nodes[elements[1,2], :] 
    n4 = nodes[elements[1,4], :] 
    n5 = nodes[elements[1,5], :] 
    
    dx = norm(n2 - n1)
    dy = norm(n4 - n1)
    dz = norm(n5 - n1)
    
    Ke_base = Element.get_canonical_stiffness(dx, dy, dz, nu)
    canonical_time = time() - canonical_start
    
    @printf("  Element size:          %.6f × %.6f × %.6f m\n", dx, dy, dz)
    @printf("  Canonical K computed:  %.3f ms\n", canonical_time * 1000)
    println()

    # Pre-allocate arrays
    entries_per_elem = 576
    total_entries = nActive * entries_per_elem
    
    estimated_mem = total_entries * (4 + 4 + 4) / 1024^2  # 3 arrays: I, J, V (Int32, Int32, Float32)
    @printf("  Matrix entries:        %12d\n", total_entries)
    @printf("  Estimated memory:      %12.2f MB\n", estimated_mem)
    println()

    alloc_start = time()
    I_vec = Vector{Int32}(undef, total_entries)
    J_vec = Vector{Int32}(undef, total_entries)
    V_vec = Vector{T}(undef, total_entries)
    alloc_time = time() - alloc_start

    @printf("  Memory allocated:      %.3f ms\n", alloc_time * 1000)
    println("  Assembling element contributions (parallel)...")
    
    # Parallel assembly
    thread_assembly_start = time()
    
    Threads.@threads for t_idx in 1:length(active_indices)
        e = active_indices[t_idx]
        offset = (t_idx - 1) * entries_per_elem
        factor = E * density[e]
        conn = view(elements, e, :)
        
        cnt = 0
        @inbounds for i in 1:8
            row_node = conn[i]
            row_base_dof = 3 * (row_node - 1)
            
            for r_dof in 1:3
                global_row = row_base_dof + r_dof
                local_row = 3*(i-1) + r_dof
                
                for j in 1:8
                    col_node = conn[j]
                    col_base_dof = 3 * (col_node - 1)
                    
                    for c_dof in 1:3
                        global_col = col_base_dof + c_dof
                        local_col = 3*(j-1) + c_dof
                        
                        cnt += 1
                        idx = offset + cnt
                        
                        I_vec[idx] = Int32(global_row)
                        J_vec[idx] = Int32(global_col)
                        V_vec[idx] = Ke_base[local_row, local_col] * factor
                    end
                end
            end
        end
    end
    
    thread_assembly_time = time() - thread_assembly_start
    @printf("  Assembly completed:    %.3f s (%.1f elem/s)\n", 
            thread_assembly_time, nActive / thread_assembly_time)
    println()

    # Build sparse matrix
    sparse_start = time()
    K_cpu = sparse(I_vec, J_vec, V_vec, ndof, ndof)
    sparse_time = time() - sparse_start
    
    @printf("  Sparse matrix built:   %.3f s\n", sparse_time)
    
    # Symmetrize
    symmetrize_start = time()
    K_cpu = (K_cpu + K_cpu') / T(2.0)
    symmetrize_time = time() - symmetrize_start
    
    @printf("  Symmetrized:           %.3f s\n", symmetrize_time)
    
    # Matrix statistics
    nnz_K = nnz(K_cpu)
    sparsity = 100.0 * (1.0 - nnz_K / (ndof * ndof))
    @printf("  Non-zeros:             %12d\n", nnz_K)
    @printf("  Sparsity:              %12.2f%%\n", sparsity)
    println()
    
    diag_K = diag(K_cpu)
    
    total_assembly_time = time() - assembly_start
    @printf("  TOTAL ASSEMBLY TIME:   %.3f s\n", total_assembly_time)
    println("="^80)
    println()

    return K_cpu, diag_K
end

# ============================================================================
# NATIVE GPU CG SOLVER - IMPROVED
# ============================================================================

function gpu_sparse_cg_solve(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                             bc_indicator::Matrix{T}, f::Vector{T},
                             density::Vector{T};
                             max_iter=1000, tol=1e-6,
                             shift_factor::T=Float32(1.0e-6),
                             min_stiffness_threshold::T=Float32(1.0e-3),
                             config::Dict=Dict()) where T
                             
    total_solve_start = time()
    
    CUDA.allowscalar(false)
    
    # Assemble system
    K_cpu, diag_K = assemble_sparse_matrix_optimized(nodes, elements, E, nu, density, min_stiffness_threshold)
    
    ndof = size(K_cpu, 1)
    free_dofs = get_free_dofs(bc_indicator)
    
    # Get current iteration from config if available
    current_iter = get(config, "current_iteration", 0)
    if current_iter > 0
        print_section_header("GPU SOLVER - NATIVE CG (ITERATION $current_iter)")
    else
        print_section_header("GPU SOLVER - NATIVE CG")
    end
    
    @printf("  Free DOFs:             %12d / %d (%.1f%%)\n", 
            length(free_dofs), ndof, 100.0 * length(free_dofs) / ndof)
    println()
    
    # Reduce system
    reduction_start = time()
    K_free_cpu = K_cpu[free_dofs, free_dofs]
    f_free_cpu = f[free_dofs]
    reduction_time = time() - reduction_start
    
    @printf("  System reduction:      %.3f s\n", reduction_time)
    
    # Apply diagonal shift
    try
        max_diag = maximum(abs.(diag(K_free_cpu)))
        min_diag = minimum(abs.(diag(K_free_cpu)))
        cond_estimate = max_diag / max(min_diag, 1e-12)
        
        shift = shift_factor * max_diag
        @printf("  Max diagonal:          %.6e\n", max_diag)
        @printf("  Min diagonal:          %.6e\n", min_diag)
        @printf("  Condition estimate:    %.6e\n", cond_estimate)
        @printf("  Diagonal shift:        %.6e (factor: %.1e)\n", shift, shift_factor)
        
        K_free_cpu = K_free_cpu + shift * I
    catch e
        @warn "Could not apply diagonal shift: $e"
    end
    println()

    # Transfer to GPU
    gpu_transfer_start = time()
    A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(K_free_cpu)
    b_gpu = CuVector(f_free_cpu)
    x_gpu = CUDA.zeros(T, length(f_free_cpu))
    gpu_transfer_time = time() - gpu_transfer_start

    @printf("  GPU transfer:          %.3f s\n", gpu_transfer_time)
    
    # Check for zero force
    norm_b = norm(b_gpu)
    @printf("  Force norm:            %.6e\n", norm_b)
    
    if norm_b == 0
        println("\n⚠️  Zero force vector. Solution is zero.")
        println("="^80)
        x_full = zeros(T, ndof)
        return x_full
    end
    println()

    # CG iteration setup with vector recycling
    println("  Starting CG iterations (printing every 1000 iterations)...")
    println("  " * "-"^66)
    @printf("  %8s %16s %16s %12s\n", "Iter", "Residual", "Rel. Residual", "Time (s)")
    println("  " * "-"^66)
    
    # Pre-allocate vectors for recycling (avoid repeated allocations)
    r_gpu = CUDA.zeros(T, length(b_gpu))
    p_gpu = CUDA.zeros(T, length(b_gpu))
    Ap_gpu = CUDA.zeros(T, length(b_gpu))
    
    # Initial residual: r = b - A*x
    CUDA.CUSPARSE.mv!('N', T(1.0), A_gpu, x_gpu, T(0.0), r_gpu, 'O')  # r = A*x
    r_gpu .= b_gpu .- r_gpu  # r = b - r
    
    # Initial search direction
    p_gpu .= r_gpu
    
    rsold_gpu = dot(r_gpu, r_gpu)
    
    cg_start = time()
    converged = false
    final_iter = 0
    
    # Print interval: every 1000 iterations
    print_interval = 1000

    for iter in 1:max_iter
        # Matrix-vector product: Ap = A*p (reuse Ap_gpu buffer)
        CUDA.CUSPARSE.mv!('N', T(1.0), A_gpu, p_gpu, T(0.0), Ap_gpu, 'O')
        
        denom = dot(p_gpu, Ap_gpu)
        
        if abs(denom) < 1e-20
            @warn "  CG breakdown: denominator too small at iteration $iter"
            break
        end

        alpha = rsold_gpu / denom
        
        # Update solution: x = x + alpha*p
        CUDA.CUBLAS.axpy!(length(x_gpu), alpha, p_gpu, 1, x_gpu, 1)
        
        # Update residual: r = r - alpha*Ap
        CUDA.CUBLAS.axpy!(length(r_gpu), -alpha, Ap_gpu, 1, r_gpu, 1)
        
        rsnew_gpu = dot(r_gpu, r_gpu)
        residual_norm = sqrt(rsnew_gpu) / norm_b
        
        # Print progress: first iteration, every 1000 iterations, or on convergence
        should_print = (iter == 1) || (iter % print_interval == 0) || (residual_norm < tol)
        
        if should_print
            elapsed = time() - cg_start
            @printf("  %8d %16.8e %16.8e %12.3f\n", iter, sqrt(rsnew_gpu), residual_norm, elapsed)
        end
        
        # Check convergence
        if residual_norm < tol
            converged = true
            final_iter = iter
            # Print final iteration if it wasn't already printed
            if !should_print
                elapsed = time() - cg_start
                @printf("  %8d %16.8e %16.8e %12.3f\n", iter, sqrt(rsnew_gpu), residual_norm, elapsed)
            end
            break
        end
        
        # Update search direction: p = r + beta*p
        beta = rsnew_gpu / rsold_gpu
        CUDA.CUBLAS.scal!(length(p_gpu), beta, p_gpu, 1)  # p = beta*p
        CUDA.CUBLAS.axpy!(length(p_gpu), T(1.0), r_gpu, 1, p_gpu, 1)  # p = r + p
        
        rsold_gpu = rsnew_gpu
        final_iter = iter
    end
    
    cg_time = time() - cg_start
    println("  " * "-"^66)
    println()
    
    # Summary
    if converged
        @printf("  ✓ CONVERGED in %d iterations\n", final_iter)
    else
        @printf("  ⚠️  DID NOT CONVERGE (max iterations reached)\n")
    end
    @printf("  CG solve time:         %.3f s\n", cg_time)
    @printf("  Avg time/iteration:    %.3f ms\n", 1000.0 * cg_time / final_iter)
    println()

    # Transfer back
    transfer_back_start = time()
    x_free = Array(x_gpu)
    transfer_back_time = time() - transfer_back_start
    
    @printf("  GPU → CPU transfer:    %.3f ms\n", transfer_back_time * 1000)
    
    # Reconstruct full solution
    x_full = zeros(T, ndof)
    x_full[free_dofs] = x_free
    
    total_solve_time = time() - total_solve_start
    
    println()
    @printf("  TOTAL SOLVE TIME:      %.3f s\n", total_solve_time)
    println("="^80)
    println()
    
    return x_full
end

# ============================================================================
# KRYLOV GPU SOLVER - IMPROVED
# ============================================================================

function gpu_krylov_solve(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T},
                          density::Vector{T};
                          solver=:cg, max_iter=1000, tol=1e-6, use_precond=true,
                          shift_factor::T=Float32(1.0e-6),
                          min_stiffness_threshold::T=Float32(1.0e-3),
                          config::Dict=Dict()) where T
                            
    total_solve_start = time()
    
    CUDA.allowscalar(false)

    # Assemble system
    K_cpu, diag_K = assemble_sparse_matrix_optimized(nodes, elements, E, nu, density, min_stiffness_threshold)
    
    ndof = size(K_cpu, 1)
    free_dofs = get_free_dofs(bc_indicator)
    
    # Get current iteration from config if available
    current_iter = get(config, "current_iteration", 0)
    if current_iter > 0
        print_section_header("GPU SOLVER - KRYLOV.$solver (ITERATION $current_iter)")
    else
        print_section_header("GPU SOLVER - KRYLOV.$solver")
    end
    
    @printf("  Free DOFs:             %12d / %d (%.1f%%)\n", 
            length(free_dofs), ndof, 100.0 * length(free_dofs) / ndof)
    @printf("  Preconditioner:        %12s\n", use_precond ? "Jacobi" : "None")
    println()
    
    # Reduce system
    reduction_start = time()
    K_free_cpu = K_cpu[free_dofs, free_dofs]
    f_free_cpu = f[free_dofs]
    diag_free_cpu = diag_K[free_dofs]
    reduction_time = time() - reduction_start
    
    @printf("  System reduction:      %.3f s\n", reduction_time)
    
    # Apply diagonal shift
    try
        max_diag = maximum(abs.(diag(K_free_cpu)))
        min_diag = minimum(abs.(diag(K_free_cpu)))
        cond_estimate = max_diag / max(min_diag, 1e-12)
        
        shift = shift_factor * max_diag
        @printf("  Max diagonal:          %.6e\n", max_diag)
        @printf("  Min diagonal:          %.6e\n", min_diag)
        @printf("  Condition estimate:    %.6e\n", cond_estimate)
        @printf("  Diagonal shift:        %.6e (factor: %.1e)\n", shift, shift_factor)
        
        K_free_cpu = K_free_cpu + shift * I
        diag_free_cpu .+= shift
    catch e
        @warn "Could not apply diagonal shift: $e"
    end
    println()

    # Transfer to GPU
    gpu_transfer_start = time()
    A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(K_free_cpu)
    b_gpu = CuVector(f_free_cpu)
    gpu_transfer_time = time() - gpu_transfer_start

    @printf("  GPU transfer:          %.3f s\n", gpu_transfer_time)
    
    # Check for zero force
    norm_b = norm(b_gpu)
    @printf("  Force norm:            %.6e\n", norm_b)
    
    if norm_b == 0
        println("\n⚠️  Zero force vector. Solution is zero.")
        println("="^80)
        x_full = zeros(T, ndof)
        return x_full
    end
    println()
    
    # Setup preconditioner
    opM = nothing
    if use_precond
        precond_start = time()
        
        # Ensure positive definite diagonal
        replace!(x -> x < 1e-12 ? T(1.0) : x, diag_free_cpu)
        d_gpu = CuVector(diag_free_cpu)
        
        function ldiv_jacobi!(y, x, d_gpu)
            y .= x ./ d_gpu
            return y
        end
        
        symmetric = true
        hermitian = true
        n = length(b_gpu)
        
        opM = LinearOperator(T, n, n, symmetric, hermitian, 
                             (y, x) -> ldiv_jacobi!(y, x, d_gpu))
        
        precond_time = time() - precond_start
        @printf("  Preconditioner setup:  %.3f ms\n", precond_time * 1000)
        println()
    end

    # Solve with minimal verbosity
    # Note: Krylov.jl callback API varies by version, so we use verbose=0 for clean output
    println("  Starting Krylov.$solver iterations...")
    println("  (Progress updates every 1000 iterations will appear below)")
    println()
    
    solve_start = time()
    iteration_count = Ref(0)
    last_print = Ref(0)
    print_interval = 1000
    
    # Create simple progress tracking that works with Krylov's internal iteration counter
    # We'll monitor the stats object after solve completes
    
    # Solve with verbose=0 (completely quiet)
    x_gpu, stats = if opM !== nothing
        if solver == :cg
            cg(A_gpu, b_gpu, M=opM, itmax=max_iter, rtol=tol, verbose=0, history=true)
        elseif solver == :minres
            minres(A_gpu, b_gpu, M=opM, itmax=max_iter, rtol=tol, verbose=0, history=true)
        elseif solver == :bicgstab
            bicgstab(A_gpu, b_gpu, M=opM, itmax=max_iter, rtol=tol, verbose=0, history=true)
        else
            error("Unknown Krylov solver: $solver")
        end
    else
        if solver == :cg
            cg(A_gpu, b_gpu, itmax=max_iter, rtol=tol, verbose=0, history=true)
        elseif solver == :minres
            minres(A_gpu, b_gpu, itmax=max_iter, rtol=tol, verbose=0, history=true)
        elseif solver == :bicgstab
            bicgstab(A_gpu, b_gpu, itmax=max_iter, rtol=tol, verbose=0, history=true)
        else
            error("Unknown Krylov solver: $solver")
        end
    end
    
    solve_time = time() - solve_start
    
    # Print iteration history table
    println()
    
    # Try different field names for residual history (varies by Krylov version)
    residuals = nothing
    if hasproperty(stats, :residuals) && !isnothing(stats.residuals) && length(stats.residuals) > 0
        residuals = stats.residuals
        println("  (Using stats.residuals)")
    elseif hasproperty(stats, :Residuals) && !isnothing(stats.Residuals) && length(stats.Residuals) > 0
        residuals = stats.Residuals
        println("  (Using stats.Residuals)")
    elseif hasproperty(stats, :rNorms) && !isnothing(stats.rNorms) && length(stats.rNorms) > 0
        residuals = stats.rNorms
        println("  (Using stats.rNorms)")
    else
        # Debug: show what fields are available
        println("  Available stats fields: ", fieldnames(typeof(stats)))
    end
    
    if residuals !== nothing && length(residuals) > 0
        println("  " * "-"^66)
        @printf("  %8s %16s %16s %12s\n", "Iter", "Residual", "Rel. Residual", "Time (s)")
        println("  " * "-"^66)
        
        # Print first iteration
        @printf("  %8d %16.8e %16.8e %12.3f\n", 
                1, residuals[1], residuals[1]/norm_b, 0.0)
        
        # Print every 1000th iteration from history
        for i in 1000:1000:length(residuals)
            elapsed_est = solve_time * (i / length(residuals))
            @printf("  %8d %16.8e %16.8e %12.3f\n", 
                    i, residuals[i], residuals[i]/norm_b, elapsed_est)
        end
        
        # Get final iteration count
        niter_final = length(residuals)
        
        # Print final iteration if not already printed
        if niter_final % 1000 != 0
            @printf("  %8d %16.8e %16.8e %12.3f\n", 
                    niter_final, residuals[end], residuals[end]/norm_b, solve_time)
        end
        println("  " * "-"^66)
    else
        @warn "Residual history not available - enable with history=true"
    end
    
    println()
    
    # Summary
    if stats.solved
        @printf("  ✓ CONVERGED\n")
    else
        @printf("  ⚠️  DID NOT CONVERGE\n")
    end
    
    # Get iteration count safely
    niter = if hasproperty(stats, :niter)
        stats.niter
    elseif residuals !== nothing
        length(residuals)
    else
        0
    end
    
    @printf("  Iterations:            %12d / %d\n", niter, max_iter)
    
    if residuals !== nothing && length(residuals) > 0
        @printf("  Initial residual:      %.6e\n", residuals[1])
        @printf("  Final residual:        %.6e\n", residuals[end])
        @printf("  Relative residual:     %.6e\n", residuals[end] / norm_b)
    end
    
    @printf("  Solve time:            %.3f s\n", solve_time)
    if niter > 0
        @printf("  Avg time/iteration:    %.3f ms\n", 1000.0 * solve_time / niter)
    end
    println()

    # Transfer back
    transfer_back_start = time()
    x_free = Array(x_gpu)
    transfer_back_time = time() - transfer_back_start
    
    @printf("  GPU → CPU transfer:    %.3f ms\n", transfer_back_time * 1000)
    
    # Reconstruct full solution
    x_full = zeros(T, ndof)
    x_full[free_dofs] = x_free
    
    total_solve_time = time() - total_solve_start
    
    println()
    @printf("  TOTAL SOLVE TIME:      %.3f s\n", total_solve_time)
    println("="^80)
    println()

    return x_full
end

# ============================================================================
# HIGH-LEVEL INTERFACE
# ============================================================================

function solve_system_gpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T},
                          density::Vector{T};
                          max_iter=1000, tol=1e-6, 
                          method=:native, solver=:cg, use_precond=true,
                          shift_factor::T=Float32(1.0e-6),
                          min_stiffness_threshold::T=Float32(1.0e-3),
                          config::Dict=Dict()) where T
                            
    @assert density !== nothing "You must provide a density array for GPU solver."
    
    if !CUDA.functional()
        error("CUDA is not functional on this system.")
    end
    
    println("\n")
    println("╔" * "═"^68 * "╗")
    println("║" * " "^20 * "GPU FEM SOLVER INITIATED" * " "^24 * "║")
    println("╚" * "═"^68 * "╝")
    println()
    
    # GPU info
    gpu_info_start = time()
    free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory()
    @printf("  GPU Device:            %s\n", CUDA.name(CUDA.device()))
    @printf("  Available memory:      %.2f GB / %.2f GB\n", 
            free_mem / 1024^3, total_mem / 1024^3)
    @printf("  Method:                %s\n", method == :native ? "Native CG" : "Krylov.$solver")
    @printf("  Tolerance:             %.2e\n", tol)
    @printf("  Max iterations:        %d\n", max_iter)
    println()
    
    overall_start = time()
    
    solution = if method == :native
        gpu_sparse_cg_solve(nodes, elements, E, nu, bc_indicator, f, density,
                           max_iter=max_iter, tol=tol, shift_factor=shift_factor,
                           min_stiffness_threshold=min_stiffness_threshold,
                           config=config)
    elseif method == :krylov
        gpu_krylov_solve(nodes, elements, E, nu, bc_indicator, f, density,
                        solver=solver, max_iter=max_iter, tol=tol, 
                        use_precond=use_precond, shift_factor=shift_factor,
                        min_stiffness_threshold=min_stiffness_threshold,
                        config=config)
    else
        error("Unknown method: $method. Use :native or :krylov.")
    end
    
    overall_time = time() - overall_start
    
    println("╔" * "═"^68 * "╗")
    println("║" * " "^20 * "GPU SOLVER COMPLETED" * " "^28 * "║")
    println("╠" * "═"^68 * "╣")
    @printf("║  Total time: %48.3f s  ║\n", overall_time)
    println("╚" * "═"^68 * "╝")
    println()
    
    return solution
end

end