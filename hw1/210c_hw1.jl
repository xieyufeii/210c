# Yufei Xie, yux085@ucsd.edu

using Plots, Parameters, SparseArrays, LinearAlgebra, Statistics, Roots, ShiftedArrays
using Distributions, Expectations, Plots, BenchmarkTools, DelimitedFiles, Random

β = 0.99;
κ = 0.13;
ρ = 0.8;
σ_ϵ = 0.00066;
σ_η = 0.007;

a = 0.0;
b = ρ*(1-β*ρ)/(1-β*ρ+κ);
c = (1-β*ρ)/(1-β*ρ+κ);
d = κ/(1+κ);

T = 500;
N = 500;
H = 20;

Random.seed!(1111)

function simulate_model()
    ΔM = zeros(T);
    y = zeros(T);
    η = σ_η.*randn(T);
    ε = σ_ϵ.*randn(T);
    for t in 2:T
        ΔM[t] = ρ*ΔM[t-1]+ε[t];
        y[t] = a*y[t-1] + b*ΔM[t-1] + c*ε[t] + d*η[t];
    end 
    return y, ΔM, ε
end


function lag(x::Vector, k::Int)
    return [fill(NaN, k); x[1:end-k]]
end


function estimate_coef(y, ε, lagsy, lagsϵ)
    if lagsϵ == 0
        X = hcat(ones(T,1),[lag(y, l) for l in 1:lagsy]..., ε);
    else
        X = hcat(ones(T,1),[lag(y, l) for l in 1:lagsy]..., [lag(ε, l) for l in 1:lagsϵ]..., ε);
    end
    X = X[max(lagsy,lagsϵ)+1:end, :];
    y_trim = y[max(lagsy,lagsϵ)+1:end];
    β̂ = X \ y_trim; # constant, lagy, lagϵ, ε
    # β̂ = inv(X'X)*X'y_trim; # constant, lagy, lagϵ, ε - slow
    return β̂
end

# true irf
function irf_truemodel()
    irf_sim = zeros(H);
    for h in 1:H
        irf_sim[h] = c*ρ^(h-1);
    end
    return irf_sim
end
irf_true = irf_truemodel();

function irf_1lagy(y, ε, H)
    β̂ = estimate_coef(y, ε, 1, 0);
    ϵ_sim = zeros(H);
    y_sim = zeros(H);
    irf_sim = zeros(H);
    ϵ_sim[1] = 1.0;
    y_sim[1] = β̂[1] + β̂[3]*ϵ_sim[1];
    for t in 2:H
        y_sim[t] = β̂[1] + β̂[2]*y_sim[t-1] + β̂[3]*ε[t];
    end
    irf_sim = y_sim[1:H] .- β̂[1];
    return irf_sim
end

function irf_4lagy(y, ε, H)
    β̂ = estimate_coef(y, ε, 4, 0);
    ϵ_sim = zeros(H);
    y_sim = zeros(H);
    irf_sim = zeros(H);
    ϵ_sim[1] = 1.0;
    for t in 1:H
        y_sim[t] = β̂[1]
        for j in 1:4
            if t - j > 0
                y_sim[t] += β̂[j+1]*y_sim[t-j]
            end
        end
        y_sim[t] += β̂[end]*ϵ_sim[t]
    end
    irf_sim = y_sim[1:H] .- β̂[1];
    return irf_sim
end

function irf_12lagy(y, ε, H)
    β̂ = estimate_coef(y, ε, 12, 0);
    ϵ_sim = zeros(H);
    y_sim = zeros(H);
    irf_sim = zeros(H);
    ϵ_sim[1] = 1.0;
    for t in 1:H
        y_sim[t] = β̂[1]
        for j in 1:12
            if t - j > 0
                y_sim[t] += β̂[j+1]*y_sim[t-j]
            end
        end
        y_sim[t] += β̂[end]*ϵ_sim[t]
    end
    irf_sim = y_sim[1:H] .- β̂[1];
    return irf_sim
end

function irf_1lagy6lagϵ(y, ε, H)
    β̂ = estimate_coef(y, ε, 1, 6);
    ϵ_sim = zeros(H);
    y_sim = zeros(H);
    irf_sim = zeros(H);
    ϵ_sim[1] = 1.0;
    for t in 1:H
        y_sim[t] = β̂[1]
        for j in 1:6
            if t - j > 0
                y_sim[t] += β̂[j+2] * ϵ_sim[t-j]
            end
        end
        if t > 1
            y_sim[t] += β̂[2]*y_sim[t-1] + β̂[end]*ϵ_sim[t]
        else
            y_sim[t] += β̂[end]*ϵ_sim[t]
        end
    end
    irf_sim = y_sim[1:H] .- β̂[1];
    return irf_sim
end

function irf_jorda(y, ΔM, ε, H)
    irf_sim = zeros(H);
    irfc_sim = zeros(H);
    for j in 0:H-1
        y_trim = y[2+j:T] .- y[1:T-j-1];
        ε_trim = ε[2:T-j];
        y_control = y[1:T-j-1];
        ΔM_control = ΔM[1:T-j-1];
        X = hcat(ones(T-j-1,1), ε_trim);
        Xc = hcat(ones(T-j-1,1), ε_trim, y_control, ΔM_control); # control for y_t-1, ΔM_t-1
        β̂ = X \ y_trim;
        β̂c = Xc \ y_trim;
        irf_sim[1+j] = β̂[2];
        irfc_sim[1+j] = β̂c[2];
    end
    return irf_sim,irfc_sim
end

# simulation
irf1y_all = zeros(H,N);
irf4y_all = zeros(H,N);
irf12y_all = zeros(H,N);
irf1y6ϵ_all = zeros(H,N);
irfjorda_all = zeros(H,N);
irfjordac_all = zeros(H,N);
for s in 1:N
    y, ΔM, ε = simulate_model();
    irf1y_all[:,s] = irf_1lagy(y, ε, H);
    irf4y_all[:,s] = irf_4lagy(y, ε, H);
    irf12y_all[:,s] = irf_12lagy(y, ε, H);
    irf1y6ϵ_all[:,s] = irf_1lagy6lagϵ(y, ε, H);
    irfjorda_all[:,s],irfjordac_all[:,s] = irf_jorda(y, ΔM, ε, H);
end

# plot median
irf1y_plot = median(irf1y_all, dims=2);
irf4y_plot = median(irf4y_all, dims=2);
irf12y_plot = median(irf12y_all, dims=2);
irf1y6ϵ_plot = median(irf1y6ϵ_all, dims=2);
irfjorda_plot = median(irfjorda_all, dims=2);
irfjordac_plot = median(irfjordac_all, dims=2);

## problem 3
plot(1:H, irf1y_plot, label="irf 1 lagy", xlabel="t", lw=2, size=(500, 350))
plot!(1:H, irf4y_plot, label="irf 4 lagy", xlabel="t", lw=2)
plot!(1:H, irf12y_plot, label="irf 12 lagy", xlabel="t", lw=2)
plot!(1:H, irf_true, label="irf true", xlabel="t", lw=2, color=:black, linestyle = :dash)
savefig("./p3.svg")

## problem 4
plot(1:H, irf1y6ϵ_plot, label="irf 1 lagy 6 lagϵ", xlabel="t", lw=2, size=(500, 350))
plot!(1:H, irf_true, label="irf true", xlabel="t", lw=2, color=:black,linestyle = :dash)
savefig("./p4.svg")

## problem 5: Jorda specification
plot(1:H, irfjorda_plot, label="irf jorda wo control", xlabel="t", lw=2,color=:red, size=(500, 350))
plot!(1:H, irfjordac_plot, label="irf jorda w control", xlabel="t", lw=2,color=:green, linestyle = :dash)
plot!(1:H, irf_true, label="irf true", xlabel="t", lw=2,color=:black, linestyle = :dash)
savefig("./p5.svg")

