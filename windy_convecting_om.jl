using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf
using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity
    
#####
##### Model setup
#####

N² = 1e-5  # Buoyancy frequency
Qᵀ = 0.0 # what should this be?
Qᵘ = -1e-3 #
α = 2e-4 # thermal expansion coefficient
T₀ = 20.0 # surface temperature
f = 1e-4

# Set up a single column grid
grid = RectilinearGrid(size=200, z=(-1000, 0), topology=(Flat, Flat, Bounded))

coriolis = FPlane(; f)

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

closure = RiBasedVerticalDiffusivity()

equation_of_state = LinearEquationOfState(thermal_expansion=α)
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35.0)

# Nutrient, plankton, detritis, bacteria model
field_dependencies = (:N, :P, :D, :B)

parameters = (μ₀ = 1/day,
              λ  = 20.0,
              m  = 1.0,
              I₀ = 700.0, # W m⁻²
              kᴵ = 10.0,
              kᴺ = 0.1,
              kᴰ = 0.1,
              y  = 0.3) # yield

@inline I(z, λ, I₀) = I₀ * exp(z / λ)
@inline light(z, λ, I₀, kᴵ) = I(z, λ, I₀) / (I(z, λ, I₀) + kᴵ)
@inline P_growth(z, N, P, p) = p.μ₀ * light(z, p.λ, p.I₀, p.kᴵ) * N / (N + p.kᴺ) * P
@inline B_growth(z, D, B, p) = p.μ₀ * D / (D + p.kᴰ) * B

# Plankton dynamics
@inline P_rhs(x, y, z, t, N, P, D, B, p) = + P_growth(z, N, P, p) - p.m * P^2

# Bacteria dynamics
@inline B_rhs(x, y, z, t, N, P, D, B, p) = + B_growth(z, D, B, p) - p.m * B^2

# Detritus dynamics
@inline D_rhs(x, y, z, t, N, P, D, B, p) = - B_growth(z, D, B, p) / p.y + p.m * B^2 + p.m * P^2

# Nutrient dynamics
@inline N_rhs(x, y, z, t, N, P, D, B, p) = - P_growth(z, N, P, p) + (1 / p.y - 1) * B_growth(z, D, B, p)

forcing = (;
    P =  Forcing(P_rhs; field_dependencies, parameters),
    B =  Forcing(B_rhs; field_dependencies, parameters),
    N =  Forcing(P_rhs; field_dependencies, parameters),
    D = (Forcing(P_rhs; field_dependencies, parameters), AdvectiveForcing(WENO(), w=0.0)) #10/day))
)

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis, buoyancy, forcing,
                                    tracers = (:T, :N, :P, :D, :B),
                                    boundary_conditions = (; T=T_bcs, u=u_bcs))
                                    
g = buoyancy.gravitational_acceleration
Tᵢ(x, y, z) = T₀ + N² * z / (α * g)

Nᵢ(x, y, z) = 10.0
Pᵢ(x, y, z) = 0.1
Dᵢ(x, y, z) = 0.1
Bᵢ(x, y, z) = 0.1

set!(model, T=Tᵢ, N=Nᵢ, P=Pᵢ, D=Dᵢ, B=Bᵢ)

#####
##### Build a simulation
#####

#simulation = Simulation(model, Δt=20minutes, stop_time=4days)
simulation = Simulation(model, Δt=1e-2, stop_iteration=10)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                     schedule = IterationInterval(1), #TimeInterval(10minutes),
                     filename = "windy_convecting_om.jld2",
                     overwrite_existing = true)

progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

@info "Running a simulation of $model..."

run!(simulation)

#####
##### Visualize
#####

filepath = "windy_convecting_om.jld2"
T_ts = FieldTimeSeries(filepath, "T")
u_ts = FieldTimeSeries(filepath, "u")
v_ts = FieldTimeSeries(filepath, "v")
N_ts = FieldTimeSeries(filepath, "N")
P_ts = FieldTimeSeries(filepath, "P")
D_ts = FieldTimeSeries(filepath, "D")
B_ts = FieldTimeSeries(filepath, "B")

z = znodes(T_ts)
Nt = length(T_ts.times)

fig = Figure(resolution=(1200, 800))

slider = Slider(fig[2, 1:4], range=1:Nt, startvalue=1)
n = slider.value

temperature_label = @lift "Temperature at t = " * prettytime(T_ts.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(T_ts.times[$n])
nutrients_label = @lift "Nutrient concentration at t = " * prettytime(T_ts.times[$n])
pdb_label = @lift "Other stuff at t = " * prettytime(T_ts.times[$n])
ax_T = Axis(fig[1, 1], xlabel=temperature_label, ylabel="z")
ax_u = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z")
ax_N = Axis(fig[1, 3], xlabel=nutrients_label, ylabel="z")
ax_PDB = Axis(fig[1, 4], xlabel=pdb_label, ylabel="z")

colors = [:black, :blue, :red, :orange]

Tn = @lift interior(T_ts[$n], 1, 1, :)
un = @lift interior(u_ts[$n], 1, 1, :)
vn = @lift interior(v_ts[$n], 1, 1, :)
Nn = @lift interior(N_ts[$n], 1, 1, :)
Pn = @lift interior(P_ts[$n], 1, 1, :)
Dn = @lift interior(D_ts[$n], 1, 1, :)
Bn = @lift interior(B_ts[$n], 1, 1, :)

lines!(ax_T, Tn, z)
lines!(ax_u, un, z, label="u")
lines!(ax_u, vn, z, label="v")
lines!(ax_N, Nn, z)
lines!(ax_PDB, Pn, z, label="P")
lines!(ax_PDB, Dn, z, label="D")
lines!(ax_PDB, Bn, z, label="B")

axislegend(ax_u, position=:rb)
axislegend(ax_PDB, position=:rb)

display(fig)

record(fig, "windy_convecting_om.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

