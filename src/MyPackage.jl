module MyPackage


using StaticArrays
using Base.Threads
using Random, Distributions
using BenchmarkTools
using StatsBase
using CUDA

export model

function noise_Poms(f,P,c)
    return P^2*(1e-12)^2*(1+(2*1e-3/f)^4)*(2*pi*f/c)^2
end

function noise_Pacc(f,A,c)
    return A^2*(1e-15)^2*(1+(0.4*10^(-3)/f)^2)*(1+(f*1e3/8)^4)*(1/(2*pi*f))^4*(2*pi*f/c)^2
end

function P_n(f,P,A,L,c)
    return noise_Poms(f,P,c)+(3+cos(4*pi*f*L/c))*noise_Pacc(f,A,c)
end

function R(f,L,c)
    return 3/10*1/(1+0.6*(2*pi*f*L/c)^2)*(2*pi*f*L/c)^2
end

function Sn(f,L,c,P,A)
    return (P_n(f,P,A,L,c)/R(f,L,c))
end

## Ωnoise
function Omega_noise(f,L,c,P,A)
    return 4*pi^2*(3.08567758 * 1e22)^2/(3*(100*0.67*1e3)^2)*f^3*Sn(f,L,c,P,A)
end
## function for GW

#h^2 Ωgw
function Omega_gw(A_001,f,gamma)
    return 10^A_001*(f/0.001)^gamma
end
# %%
f=range(start=3*1e-5,stop=0.5, step=1e-6)
f_filtered= f[971:end]
logbins = 10 .^ range(log10(0.001), stop=log10(0.5), length=1000)
# %%
idx = [searchsortedlast(logbins, f_filtered[i]) for i in eachindex(f_filtered)]


#%%
function model(z)
    P        =  15
    A        =  3


    A_001   =   -14+z[2]*(-6+14)
    gamma   =   -3+z[1]*6
    ## n part
    stds_n=sqrt.(0.67^2*Omega_noise.(f,2.5*1e9,3*1e8,P,A))  #compute sqrt(h^2Ωnoise(fi))
    stds_n_cuda = CuArray(Float32.(stds_n))

    std_eps = CUDA.randn(Float32, length(stds_n), 94)
    CUDA.@sync samples_noise3 = stds_n_cuda .* std_eps

    std_eps = CUDA.randn(Float32, length(stds_n), 94)
    CUDA.@sync samples_noise4 = stds_n_cuda .* std_eps


    CUDA.@sync c1 = (samples_noise3.^2 + samples_noise4.^2)/2

    # omega part
    stds_omega=sqrt.(Omega_gw.(A_001,f,gamma))  #compute sqrt(h^2Ωnoise(fi))
    stds_omega_cuda = CuArray(Float32.(stds_omega))
    # repeat for each chunk
    std_eps = CUDA.randn(Float32, length(stds_omega), 94)
    CUDA.@sync samples_noise5 = stds_omega_cuda .* std_eps

    std_eps = CUDA.randn(Float32, length(stds_omega), 94)
    CUDA.@sync samples_noise6 = stds_omega_cuda .* std_eps


    CUDA.@sync c2 = (samples_noise5.^2 + samples_noise6.^2)./2

    CUDA.@sync c = c1 + c2

    Data=Array(view(mean(c, dims=2), :,1))  #mean over chunks

    return Data
end

end # module MyPackage
