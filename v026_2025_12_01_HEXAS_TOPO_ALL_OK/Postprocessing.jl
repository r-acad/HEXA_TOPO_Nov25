
module Postprocessing

using JSON, Printf
using ..Mesh
using ..ExportVTK
import MarchingCubes: MC, march

export export_iteration_results, export_smooth_watertight_stl

"""
    get_smooth_nodal_densities(density, elements, nNodes)

Converts element-based density (constant per element) into node-based density
(linear variation) by averaging the values of elements connected to each node.
"""
function get_smooth_nodal_densities(density::Vector{Float32}, elements::Matrix{Int}, nNodes::Int)
    node_sums = zeros(Float32, nNodes)
    node_counts = zeros(Int, nNodes)
    
    nElem = length(density)
    
    # Accumulate density contributions from elements to their nodes
    @inbounds for e in 1:nElem
        rho = density[e]
        for i in 1:8
            node_idx = elements[e, i]
            node_sums[node_idx] += rho
            node_counts[node_idx] += 1
        end
    end
    
    # Average
    nodal_density = zeros(Float32, nNodes)
    @inbounds for i in 1:nNodes
        if node_counts[i] > 0
            nodal_density[i] = node_sums[i] / Float32(node_counts[i])
        else
            nodal_density[i] = 0.0f0
        end
    end
    
    return nodal_density
end

"""
    trilinear_interpolate(vals, xd, yd, zd)

Interpolates value within a voxel defined by 8 corner values `vals`.
`vals` order corresponds to standard VTK/FEM hex ordering.
xd, yd, zd are normalized coordinates [0,1] within the voxel.
"""
@inline function trilinear_interpolate(vals, xd::Float32, yd::Float32, zd::Float32)
    c00 = vals[1]*(1f0-xd) + vals[2]*xd
    c01 = vals[4]*(1f0-xd) + vals[3]*xd
    c10 = vals[5]*(1f0-xd) + vals[6]*xd
    c11 = vals[8]*(1f0-xd) + vals[7]*xd

    c0 = c00*(1f0-yd) + c01*yd
    c1 = c10*(1f0-yd) + c11*yd

    return c0*(1f0-zd) + c1*zd
end

"""
    export_smooth_watertight_stl(density, geom, threshold, filename; subdivision_level=2)

Generates a high-quality, watertight STL.
1. Converts element density to nodal density.
2. Subdivides the grid (Trilinear interpolation) for smoothness.
3. Pads the grid with zeros to close boundaries.
"""
function export_smooth_watertight_stl(density::Vector{Float32}, geom, threshold::Float32, filename::String; subdivision_level::Int=2)
    println("Generating smooth watertight STL (Subdivision: $subdivision_level)...")
    
    # Ensure directory exists
    dir_path = dirname(filename)
    if !isempty(dir_path) && !isdir(dir_path)
        mkpath(dir_path)
    end

    # 1. Recover Mesh Dimensions
    NX, NY, NZ = geom.nElem_x, geom.nElem_y, geom.nElem_z
    dx, dy, dz = geom.dx, geom.dy, geom.dz
    
    # 2. Compute Smoothed Nodal Densities (Coarse)
    # We reconstruct the nodes and elements temporarily to ensure correct indexing
    nodes_coarse, elements_coarse, _ = Mesh.generate_mesh(NX, NY, NZ; dx=dx, dy=dy, dz=dz)
    nNodes_coarse = size(nodes_coarse, 1)
    
    if length(density) != size(elements_coarse, 1)
        @warn "Density vector length ($(length(density))) does not match regenerated mesh ($(size(elements_coarse, 1))). STL export might be distorted."
    end
    
    nodal_density_coarse = get_smooth_nodal_densities(density, elements_coarse, nNodes_coarse)

    # Reshape coarse data for easy indexing [i,j,k]
    # Note: Mesh.generate_mesh order is k, then j, then i
    # i=1..NX+1, j=1..NY+1, k=1..NZ+1
    grid_coarse = reshape(nodal_density_coarse, (NX+1, NY+1, NZ+1))

    # 3. Define Fine Grid Dimensions (with Padding)
    # Subdivision adds resolution. Padding (+1 layer all sides) ensures watertightness.
    sub_NX = NX * subdivision_level
    sub_NY = NY * subdivision_level
    sub_NZ = NZ * subdivision_level

    # Padded dimensions
    pad = 1 # 1 voxel padding
    fine_dim_x = sub_NX + 1 + 2*pad
    fine_dim_y = sub_NY + 1 + 2*pad
    fine_dim_z = sub_NZ + 1 + 2*pad

    # Fine Grid Spacing
    sub_dx = dx / Float32(subdivision_level)
    sub_dy = dy / Float32(subdivision_level)
    sub_dz = dz / Float32(subdivision_level)

    # 4. Generate Fine Density Grid via Interpolation
    fine_grid = zeros(Float32, fine_dim_x, fine_dim_y, fine_dim_z)
    
    # Pre-calculate coordinates for MC
    x_coords = collect(Float32, range(-pad*sub_dx, step=sub_dx, length=fine_dim_x))
    y_coords = collect(Float32, range(-pad*sub_dy, step=sub_dy, length=fine_dim_y))
    z_coords = collect(Float32, range(-pad*sub_dz, step=sub_dz, length=fine_dim_z))

    println("  Interpolating coarse field to fine grid ($(fine_dim_x)x$(fine_dim_y)x$(fine_dim_z))...")
    
    # Loop over fine grid (excluding padding, which stays 0)
    # i_f maps to physical x
    Threads.@threads for k_f in (1+pad):(fine_dim_z-pad)
        for j_f in (1+pad):(fine_dim_y-pad)
            for i_f in (1+pad):(fine_dim_x-pad)
                
                # Local index within the unpadded fine grid (0-based)
                ix = i_f - (1+pad)
                iy = j_f - (1+pad)
                iz = k_f - (1+pad)

                # --- FIX: Bounds Safe Indexing ---
                # Calculate coarse element index (0-based -> 1-based)
                idx_x = div(ix, subdivision_level)
                idx_y = div(iy, subdivision_level)
                idx_z = div(iz, subdivision_level)

                # Clamping logic: If we hit exactly the end boundary, stay in the last element
                if idx_x >= NX; idx_x = NX - 1; end
                if idx_y >= NY; idx_y = NY - 1; end
                if idx_z >= NZ; idx_z = NZ - 1; end

                # Coarse grid indices (1-based)
                c_i = idx_x + 1
                c_j = idx_y + 1
                c_k = idx_z + 1

                # Local fraction within coarse element [0, 1]
                # Note: ix goes from 0 to NX*sub.
                # relative_ix = ix - (element_start_index)
                rem_x = ix - idx_x * subdivision_level
                rem_y = iy - idx_y * subdivision_level
                rem_z = iz - idx_z * subdivision_level
                
                xd = Float32(rem_x) / Float32(subdivision_level)
                yd = Float32(rem_y) / Float32(subdivision_level)
                zd = Float32(rem_z) / Float32(subdivision_level)

                # Fetch 8 coarse corner values
                v1 = grid_coarse[c_i,   c_j,   c_k]
                v2 = grid_coarse[c_i+1, c_j,   c_k]
                v3 = grid_coarse[c_i+1, c_j+1, c_k]
                v4 = grid_coarse[c_i,   c_j+1, c_k]
                v5 = grid_coarse[c_i,   c_j,   c_k+1]
                v6 = grid_coarse[c_i+1, c_j,   c_k+1]
                v7 = grid_coarse[c_i+1, c_j+1, c_k+1]
                v8 = grid_coarse[c_i,   c_j+1, c_k+1]

                # VTK hex ordering tuple
                vals = (v1, v2, v3, v4, v5, v6, v7, v8)

                # Interpolate
                val = trilinear_interpolate(vals, xd, yd, zd)
                fine_grid[i_f, j_f, k_f] = val
            end
        end
    end

    # 5. Marching Cubes on Fine Grid
    mc_struct = MC(
        fine_grid, 
        Int; 
        normal_sign=1,
        x=x_coords, 
        y=y_coords, 
        z=z_coords
    )

    march(mc_struct, threshold)

    vertices = mc_struct.vertices
    faces = mc_struct.triangles
    
    num_vertices = length(vertices)
    num_faces = length(faces)
    
    println("  Surface extracted: $num_vertices vertices, $num_faces triangles.")
    
    if num_faces == 0
        @warn "No isosurface found at threshold $threshold. STL file will be empty/invalid."
    end

    println("  Writing Binary STL to $filename...")

    # 6. Write Binary STL (Faster and smaller than ASCII)
    try
        open(filename, "w") do io
            # Header (80 bytes) - pad with spaces
            header_str = "Binary STL generated by HEXA Topology Optimization"
            header = rpad(header_str, 80, ' ')
            write(io, header)
            
            # Number of triangles (UInt32)
            write(io, UInt32(num_faces))
            
            # Buffer for triangle data: Normal(3*4) + Vertices(3*3*4) + Attr(2) = 50 bytes
            # Pre-allocating small buffer to avoid allocating in loop? Julia IO is buffered anyway.
            
            for face in faces
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                v3 = vertices[face[3]]
                
                # Compute Normal
                e1x, e1y, e1z = v2[1]-v1[1], v2[2]-v1[2], v2[3]-v1[3]
                e2x, e2y, e2z = v3[1]-v1[1], v3[2]-v1[2], v3[3]-v1[3]
                
                nx = e1y*e2z - e1z*e2y
                ny = e1z*e2x - e1x*e2z
                nz = e1x*e2y - e1y*e2x
                
                mag = sqrt(nx*nx + ny*ny + nz*nz)
                if mag > 1e-12
                    nx /= mag; ny /= mag; nz /= mag
                else
                    nx = 0.0f0; ny = 0.0f0; nz = 0.0f0
                end
                
                # Write Normal (Float32 x 3)
                write(io, Float32(nx)); write(io, Float32(ny)); write(io, Float32(nz))
                
                # Write Vertex 1
                write(io, Float32(v1[1])); write(io, Float32(v1[2])); write(io, Float32(v1[3]))
                # Write Vertex 2
                write(io, Float32(v2[1])); write(io, Float32(v2[2])); write(io, Float32(v2[3]))
                # Write Vertex 3
                write(io, Float32(v3[1])); write(io, Float32(v3[2])); write(io, Float32(v3[3]))
                
                # Attribute byte count (UInt16) - usually 0
                write(io, UInt16(0))
            end
        end
        println("  Successfully exported STL.")
    catch e
        @error "Failed to save STL file: $e"
        println("Error details: ", e)
    end
end

function export_iteration_results(iter::Int, 
                                  base_name::String, 
                                  RESULTS_DIR::String, 
                                  nodes::Matrix{Float32}, 
                                  elements::Matrix{Int}, 
                                  U_full::Vector{Float32}, 
                                  F::Vector{Float32}, 
                                  bc_indicator::Matrix{Float32}, 
                                  principal_field::Matrix{Float32}, 
                                  vonmises_field::Vector{Float32}, 
                                  full_stress_voigt::Matrix{Float32}, 
                                  l1_stress_norm_field::Vector{Float32}, 
                                  density::Vector{Float32}, 
                                  E::Float32,
                                  geom)
      
    println("Exporting results for iteration $(iter)...") 
    iter_prefix = "iter_$(iter)_" 
    nElem = size(elements, 1) 

    # --- JSON Export (Element Data) ---
    json_filename = joinpath(RESULTS_DIR, "$(iter_prefix)$(base_name)_element_data.json") 
    element_data = [] 
    for e in 1:nElem 
        centroid_coords = Mesh.element_centroid(e, nodes, elements) 
        elem_info = Dict( 
            "element_id" => e, 
            "centroid" => centroid_coords, 
            "young_modulus" => E * density[e], 
            "von_mises_stress" => vonmises_field[e], 
            "l1_stress_norm" => l1_stress_norm_field[e], 
            "principal_stresses" => principal_field[:, e] 
        ) 
        push!(element_data, elem_info) 
    end 
      
    try 
        open(json_filename, "w") do f 
            write(f, JSON.json(element_data, 2)) 
        end 
    catch err 
        @error "Failed to write element JSON file: $err" 
    end 

    # --- VTK Exports ---
    solution_filename = joinpath(RESULTS_DIR, "$(iter_prefix)$(base_name)_solution.vtu") 
    ExportVTK.export_solution(nodes, elements, U_full, F, bc_indicator, 
                              principal_field, vonmises_field, full_stress_voigt, 
                              l1_stress_norm_field; 
                              density=density,
                              scale=Float32(1.0),  
                              filename=solution_filename) 
      
    # --- STL Export (New Smooth Logic) ---
    stl_filename = joinpath(RESULTS_DIR, "$(iter_prefix)$(base_name)_isosurface.stl")
    
    # Use a subdivision level of 2 for nice smoothing.
    # The threshold 0.3 is usually good for SIMP topologies.
    export_smooth_watertight_stl(density, geom, 0.3f0, stl_filename; subdivision_level=2)
end 

end