using Plots
using Plots.Measures

# _ Do plot headlessly
ENV["GKSwstype"] = "100"

# Favourite palettes: southwest, starrynight, seaborn_colorblind, island, sandyterrain
# _ general plots properties
gr(; framestyle=:box, fontfamily="serif-roman", size=(350 * 1.618, 350), palette=:seaborn_colorblind, margin=3mm, topmargin=1mm, xguidefontsize=10, yguidefontsize=10, xtickfontsize=10, ytickfontsize=10, legendfontsize=10, titlefontsize=10, thickness_scaling=1.00)


# __ Functions using plot
using HypothesisTests

"""
    plot_dm_test(df_res::DataFrame) -> plot_variable
Plot Diebold-Mariano results as Matrix for Models
Similar to the literature on Energy forecasting

# Example
df_losses = DataFrame(randn(100,5), "Method " .* string.(1:5))
px = plot(plot_dm_test(df_losses), size=(600,500))
"""
function plot_dm_test(df_res::DataFrame)
    K = size(df_res,2)
    pvals = ones(K, K)
    for i in 1:K
        for j in 1:K
            if i==j
                pvals[i,j] = 1.0
            else
                pvals[i,j] = pvalue(DieboldMarianoTest(df_res[!,i], df_res[!,j]; loss=abs); tail=:right)
            end
        end
    end
    pvals[pvals .>= 0.1] .= 1.0
    p_heat = plot(; legend=:outerright, xrotation=90, title="DM test", margin=1mm, framestyle=:grid)
    # p_heat = xlabel!("Performance Models A")
    # p_heat = ylabel!("Performance Models B")
    p_heat = heatmap!(pvals; clim=(0,0.10), colormap=[LinRange(colorant"green", colorant"yellow", 5); LinRange(colorant"orange", colorant"red",5); LinRange(colorant"black", colorant"black",1)], yflip=true)
    yticks!(1:K, names(df_res)) # "j ".*names(df_res)
    xticks!(1:K, names(df_res)) # "i ".*names(df_res)
    p_heat = annotate!(1:K, 1:K, ("x", "Helvetica", :white, :center, 10), label=:none, legend=:none)
    return p_heat
end
