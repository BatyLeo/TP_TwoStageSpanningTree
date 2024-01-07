### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 33d91779-378b-4e57-a779-34cf25045d2b
begin # Notebook utilities
	using PlutoTeachingTools 
	using PlutoUI
	using PlutoGrader
	using ProgressLogging
	using Test
end

# ╔═╡ e11f28b6-8f91-11ee-088d-d51c110208c6
begin
	using DataStructures # For efficient data structures
	using Flux  # gradient descent optimizers
	using GLPK  # open source MIP solver
	using Graphs # graph data handling
	using JuMP  # Mathematical programming modeling
	using LinearAlgebra  # TODO: check if it can be removed
	using Plots  # plotting tools
	using Random  # random number generator
end

# ╔═╡ 6cd3608b-0df6-4d22-af6c-9b6cb1d955c3
md"""# Two-Stage spanning tree"""

# ╔═╡ 1b51af71-8474-46e6-91b2-35fc9adb2c5a
TableOfContents()

# ╔═╡ 30670737-22c2-42e1-a0a4-43aa0fa70752
ChooseDisplayMode()

# ╔═╡ ab5c0cf3-6ce9-4908-8c09-b664503e5ac1
md"""# I - Problem statement"""

# ╔═╡ bcc75aa2-9cd7-49bf-ad46-9dd5a8db3ef0
md"""## 1. Minimum weight spanning tree (MST)"""

# ╔═╡ 4ca8f8f1-058f-4b47-adce-4cdbe916d628
md"""
- **Instance**: $x= (G, c)\in \mathcal{X}$.
  - Undirected graph: $G = (V, E)$
  - Cost function: ``c\colon e\longmapsto \mathbb{R}``.

- **Goal**: find a spanning tree over $G$ with minimum total weight

- MIP formulation:
```math
\begin{aligned}
	\min_y\quad & \sum_{e\in E} c_e y_e\\
	\text{s.t.}\quad & \sum_{e\in E} y_e = |V|-1\\
	& \sum_{e\in E(Y)} y_e \leq |Y| - 1,\quad &\forall \emptyset\subsetneq Y\subsetneq V\\
	& y_e\in \{0, 1\},\quad &\forall e\in E
\end{aligned}
```
"""

# ╔═╡ 06b5c71a-bb44-4694-be43-1b3e9b38ece2
md"""### Creating an instance"""

# ╔═╡ 2dd444e9-5df7-43c0-953c-b705bfc024a3
tip(md"""In this notebook, we use features provided by the [`Graphs.jl`](https://juliagraphs.org/Graphs.jl/stable/) library to handle graph data. Checkout its documentation for more in-depth details on its usage.

We focus on grid graphs as defined in the cell below, primarily for visualization purposes. However, note that all the techniques you will implement in the subsequent sections are applicable to any type of graph.
""")

# ╔═╡ aee968cd-1d4f-40e4-8e61-14a92bb89989
md"Create a grid graph of size `n` by `m`:"

# ╔═╡ 3f89b439-03e7-4e1e-89ab-63bbf5fa2194
n = 5; m = 4;

# ╔═╡ 0f4090e3-864c-46e5-bb28-203e735c63a8
g = Graphs.grid((n, m))

# ╔═╡ 4eab9c97-9278-4895-ba2d-1ddb78afe530
md"We generate random costs on edges, uniformly drawn in given range `c_range`:"

# ╔═╡ 5e867265-c733-485a-b39a-c4320e99c92a
begin
	c_range = 1:20
	Random.seed!(10)
	c = [rand(c_range) for _ in 1:ne(g)]
end

# ╔═╡ 21f02f67-35a2-4ff0-9343-58562d5e5bfb
md"Then, we can leverage the [`Plots.jl`](https://docs.juliaplots.org/stable/) plotting libraries to visualize the created graph and edge costs."

# ╔═╡ da2b7fef-627f-4b4a-83dc-0e731a243c61
"""
	plot_graph(graph, n, m, weights=nothing)

# Arguments
- `graph`: grid graph to plot
- `n`: n dimension
- `m`: m dimension
- `weights`: edge weights to display (optional)
"""
function plot_graph(
	graph, n, m, weights=nothing;
	show_node_indices=false, δ=0.25, δ₂=0.13,
	edge_colors=fill(:black, ne(graph)),
	edge_widths=fill(1, ne(graph)),
	edge_labels=fill(nothing, ne(graph)),
	space_for_legend=0
)
	l = [((i - 1) % n, floor((i - 1) / n)) for i in 1:nv(graph)]
	function args_from_ij(i, j)
		return [l[i][1], l[j][1]], [l[i][2], l[j][2]]
	end
	f = Plots.plot(; axis=([], false), ylimits=(-δ, m-1+δ+space_for_legend), xlimits=(-δ, n-1+δ), aspect_ratio=:equal, leg=:top)
	for (color, width, label, e) in zip(edge_colors, edge_widths, edge_labels, edges(graph))
		Plots.plot!(f, args_from_ij(src(e), dst(e)); color, width, label)
	end
	series_annotations = show_node_indices ? (1:nv(g)) : nothing
	Plots.scatter!(f, l; series_annotations, label=nothing, markersize=15, color=:lightgrey)
	if !isnothing(weights)
		for (w, e) in zip(weights, edges(graph))
			i, j = src(e), dst(e)
			x, y = (l[j] .+ l[i]) ./ 2
			if j == i + 1
				y += δ₂
			else
				x -= δ₂
			end
			Plots.annotate!(f, x, y, Int(w))
		end
	end
	return f
end

# ╔═╡ 8249ad29-f900-4992-9c32-60860d2973ee
plot_graph(g, n, m, c)

# ╔═╡ b4c0b7f5-8863-4921-915f-c7b73cb1e792
md"### Kruskal algorithm

The minimum weight spanning tree problem is polynomial, and can be solved efficiently using the kruskal algorithm:
"

# ╔═╡ b9daab11-d807-40fd-b94b-bc79ae80275e
function kruskal(g, weights; minimize=true)
    connected_vs = IntDisjointSets(nv(g))

	tree = falses(ne(g))

    edge_list = collect(edges(g))
	order = sortperm(weights; rev=!minimize)
	value = 0.0

	mst_size = 0

    for (e_ind, e) in zip(order, edge_list[order])
        if !in_same_set(connected_vs, src(e), dst(e))
            union!(connected_vs, src(e), dst(e))
			tree[e_ind] = true
			mst_size += 1
			value += weights[e_ind]
            (mst_size >= nv(g) - 1) && break
        end
    end

    return (; value, tree)
end

# ╔═╡ 77630435-3536-4714-b4c7-db4473e7ba0e
md"Other option from the `Graphs.jl` package:"

# ╔═╡ 2251158b-d21a-4e4e-bc11-89bf7c385557
Graphs.kruskal_mst

# ╔═╡ 55ca5072-6831-4794-8aed-68d8b56f7f80
tree_value, T = kruskal(g, c)

# ╔═╡ 2f9c2388-a178-452b-a013-a2cc1cabc4b4
md"""### Visualization"""

# ╔═╡ a8e889d5-a7bc-4c2e-9383-6f156eb2dd6a
"""
	plot_forest(forest, graph, n, m, weights=nothing)

# Arguments
- `forest`: forest as a BitVector
- `graph`: grid graph to plot
- `n`: n dimension
- `m`: m dimension
- `weights`: edge weights to display (optional)
"""
function plot_forest(
	forest, graph, n, m, weights=nothing; show_node_indices=false, forest_edge_width=3
)
	edge_colors = [e ? :red : :black for e in forest]
	edge_widths = [e ? forest_edge_width : 1 for e in forest]
	return plot_graph(
		graph, n, m, weights; show_node_indices, edge_colors, edge_widths
	)
end

# ╔═╡ de304db1-e5ca-4aaa-9ea7-d271bec8ae7d
plot_forest(T, g, n, m, c)

# ╔═╡ eea94e99-cc33-4464-ac24-587466b17e48
md"""The tree solution can be visualized using the `plot_forest` function:"""

# ╔═╡ c6cd42d1-c428-49a7-99a4-93f342373f06
md"## 2. Two-stage minimum weight spanning tree"

# ╔═╡ ad4284c3-a926-4c6b-8c32-4d24bcbede60
md"""
- **Instance**: Undirected graph $G = (V, E)$, scenario set $S$, $c\colon e\longmapsto \mathbb{R}$, and $d\colon(e, s)\longmapsto\mathbb{R}$.
- **Goal**: find a two stage spanning tree with minimum cost in expectation
- MIP formulation (SAA):
```math
\begin{aligned}
	\min_y\quad & \sum_{e\in E} c_e y_e + \dfrac{1}{|S|}\sum_{s\in S} d_{es} z_{es}\\
	\text{s.t.}\quad & \sum_{e\in E} (y_e + z_{es}) = |V|-1 & \forall s\in S\\
	& \sum_{e\in E(Y)} (y_e + z_{es}) \leq |Y| - 1,\quad &\forall \emptyset\subsetneq Y\subsetneq V, \forall s\in S\\
	& y_e\in \{0, 1\},\quad &\forall e\in E\\
	& z_{es}\in \{0, 1\},\quad &\forall e\in E,\,\forall s\in S
\end{aligned}
```
"""

# ╔═╡ bf369999-41c1-481f-9f17-ec7d5dd08445
md"We'll use the following `Instance` data structure:"

# ╔═╡ 7d9a2b6e-e8a9-4cf0-af4b-e45603d45008
md"""### Creating instances"""

# ╔═╡ a3eeb63a-b971-4806-9146-74936d4cc2e6
@kwdef struct Instance
    graph::SimpleGraph{Int}
    first_stage_costs::Vector{Float64}
	second_stage_costs::Matrix{Float64}
	n::Int = 0 # for plotting purposes
	m::Int = 0 # for plotting purposes
end

# ╔═╡ e14e5513-5cc2-4b70-ab29-8ee53ca166cc
nb_scenarios(instance) = size(instance.second_stage_costs, 2)

# ╔═╡ c34c3f25-58ea-4219-b856-2ed9d790d291
function random_instance(; n, m, nb_scenarios=1, c_range=1:20, d_range=1:20, seed)
	g = Graphs.grid((n, m))
	rng = MersenneTwister(seed)
	c = [rand(rng, c_range) for _ in 1:ne(g)]
	d = [rand(rng, d_range) for _ in 1:ne(g), _ in 1:nb_scenarios]

	return Instance(g, c, d, n, m)
end

# ╔═╡ b0155649-8f26-47ac-9d80-95a979f716cb
small_instance = random_instance(; n=3, m=3, nb_scenarios=1, seed=0)

# ╔═╡ c111dadd-3cb6-4cb0-b082-b67e11248e1c
S = 50

# ╔═╡ 8bc212ec-5a5d-401d-97d0-b2e0eb2b3b6f
instance = random_instance(; n, m, nb_scenarios=S, seed=0)

# ╔═╡ 8c00c839-b349-42e1-8e3f-afbd74fcf8c2
@kwdef struct Solution
	y::BitVector
	z::BitMatrix
end

# ╔═╡ d646e96c-5b2c-4349-bf11-133494af1453
function is_spanning_tree(tree_candidate::BitVector, graph::AbstractGraph)
    edge_list = [e for (i, e) in enumerate(edges(graph)) if tree_candidate[i]]
    subgraph = induced_subgraph(graph, edge_list)[1]
    return !is_cyclic(subgraph) && nv(subgraph) == nv(graph)
end

# ╔═╡ 9d2b37d1-8a73-4b3e-853a-d849b7895d01
function is_feasible(solution::Solution, instance::Instance; verbose=true)
    (; y, z) = solution
    (; graph) = instance

    # Check that no edge was selected in both stages
    if any(y .+ z .> 1)
        verbose && @warn "Same edge selected in both stages"
        return false
    end

    # Check that each scenario is a spanning tree
    S = nb_scenarios(instance)
    for s in 1:S
        if !is_spanning_tree(y .|| z[:, s], graph)
            verbose && @warn "Scenario $s is not a spanning tree: $y, $(z[:, s]), $instance"
            return false
        end
    end

    return true
end

# ╔═╡ 53a4d6de-b798-4773-830f-a26d56241b1e
function solution_from_first_stage_forest(forest::BitVector, instance::Instance)
	(; graph, second_stage_costs) = instance

	S = nb_scenarios(instance)
	forests = falses(ne(graph), S)
    for s in 1:S
		weights = deepcopy(second_stage_costs[:, s])
        m = minimum(weights) - 1
        m = min(0, m - 1)
        weights[forest] .= m  # set weights over forest as the minimum

		# find min spanning tree including forest
        _, tree_s = kruskal(graph, weights)
		forest_s = tree_s .- forest
		forests[:, s] .= forest_s
    end

    return Solution(forest, forests)
end

# ╔═╡ f81105f1-a70e-406c-ad7e-0390910e4c17
function solution_value(solution::Solution, instance::Instance)
    return dot(solution.y, instance.first_stage_costs) + dot(solution.z, instance.second_stage_costs) / nb_scenarios(instance)
end

# ╔═╡ 76cbf0da-7437-464a-ba1b-e093cabd3b83
md"""### Visualization tools"""

# ╔═╡ 6186efdf-227e-4e95-b788-5dd3219162e7
begin
	scenario_slider = @bind current_scenario PlutoUI.Slider(1:S; default=1, show_value=true);
end

# ╔═╡ 71ad5432-3c86-43da-b097-c668388b836b
function plot_scenario(
	solution::Solution, instance::Instance, scenario=current_scenario; show_node_indices=false, δ=0.25, δ₂=0.16
)
	(; graph, first_stage_costs, second_stage_costs, n, m) = instance
	first_stage_forest = solution.y
	second_stage_forests = solution.z
	
	is_labeled_1 = false
	is_labeled_2 = false
	edge_labels = fill("", ne(graph))

	S = nb_scenarios(instance)

	for e in 1:ne(graph)
		b1 = first_stage_forest[e]
		b2 = second_stage_forests[e, scenario]
		if !is_labeled_1 && b1
			edge_labels[e] = "First stage forest"
			is_labeled_1 = true
		elseif !is_labeled_2 && b2
			edge_labels[e] = "Second stage forest (scenario $scenario/$S)"
			is_labeled_2 = true
		end
	end

	edge_colors = [e1 ? :red : e2 ? :green : :black for (e1, e2) in zip(first_stage_forest, second_stage_forests[:, scenario])]
	edge_widths = [e1 || e2 ? 3 : 1 for (e1, e2) in zip(first_stage_forest, second_stage_forests[:, scenario])]
	weights = first_stage_forest .* first_stage_costs + .!first_stage_forest .* second_stage_costs[:, scenario]
	return plot_graph(
		graph, n, m, weights; show_node_indices, edge_colors, edge_widths, edge_labels, space_for_legend=3δ
	)
end

# ╔═╡ 5d2f732b-2903-45f1-aa27-4c0df5e8645b
md"# II - Branch-and-cut"

# ╔═╡ 49df95f6-34b8-48d1-b1de-40309b27c48a
md"""
The MIP formulation of the minimum weight two-stage spanning tree problem having an exponential number of constraints, we cannot solve it directly using a MIP solver, but we can solve it with a subset of constraints and iteratively add the most violated one, up until all constraint are satisfied.

Finding the most violated constraint (for scenario $s$) is called the **separation problem**, and can be formulated as:

```math
\begin{aligned}
\min_Y\quad & |Y| - 1 - \sum_{e\in E(Y)}(y_e + z_{es})\\
\text{s.t.}\quad & \emptyset \subsetneq Y\subsetneq V
\end{aligned}
```
with $y$, $z$ fixed and obtained by minimizing the problem restricted to a subset of constraints.
"""

# ╔═╡ bbedc946-cf38-4ba9-b631-0d00e5807f01
md"If the value of this problem is positive for all scenarios, then all constraints are satisfied and the optimal solution is found."

# ╔═╡ b104ec32-2b7a-42f2-99ee-6dee7c0c9cad
md"## 2. MILP separation problem formulation"

# ╔═╡ 02f6f906-c744-4e16-b73e-2dc098d6d7e3
md"""The separation problem can be formulated as the following MILP:

```math
\begin{array}{rll}
\min\limits_{\alpha, \beta}\, & \sum\limits_{v\in V}\alpha_v - 1 - \sum\limits_{e \in E} \beta_e (y_e + z_{es}) \\
\mathrm{s.t.}\, & 2 \beta_{e} \leq \alpha_u + \alpha_v \qquad & \forall e = (u,v)\in E \\
& \sum\limits_{v\in V} \alpha_v \geq 1\\
& \alpha, \beta \in \{0,1\}
\end{array}
```
"""

# ╔═╡ 16c03d54-720e-493b-bde6-34d4da9941ab
TODO(md"Implement the MIP formulation in the `MILP_separation_pb` function.

This function must have three outputs in this order
- A boolean `found` telling if a constraint is violated (i.e. if a cut should be added)
- A BitVector representing the **edges** in set ``Y`` (`Y[e] == 1` if ``e \in E(Y)``)
- The number of **vertices** in ``Y``.
")

# ╔═╡ d9441eda-d807-4452-af10-11804bc668da
function MILP_separation_problem(graph, weights; MILP_solver, tol=1e-6)
	V = nv(graph)
	E = ne(graph)

	model = Model(MILP_solver)
	set_silent(model)
	missing
end;

# ╔═╡ 3eaae7fd-8fff-43b7-9945-cdbde3b6c0fe
md"## 3. Better MILP formulation"

# ╔═╡ 7f8dea10-942e-4bae-9223-387650e35cc9
md"""
The separation problem can also be formulated as a min-cut problem, which has better performance and scaling.

The separation problem 

```math
\min  |Y| - 1 - \sum_{e \in E(Y)} (y_e + z_{es}) \quad \text{subject to} \quad \emptyset \subsetneq Y \subsetneq V
```

is equivalent to

```math
\min  |Y| + \sum_{e \notin E(Y)} (y_e + z_{es}) - |V| \quad \text{subject to} \quad \emptyset \subsetneq Y \subsetneq V.
```

Let us define the digraph ``\mathcal{D} = (\mathcal{V}, \mathcal{A})`` with vertex set ``\mathcal{V} = \{s,t\} \cup V \cup E`` and the following arcs.

| Arc ``a`` | Capacity ``u_a`` | 
| ------ | ----- |
| ``(s,e)`` for ``e \in E`` | ``y_e + z_{es}`` |
| ``(e,u)`` and ``(e,v)`` for ``e = (u,v) \in E`` | ``+\infty``|
| ``(v,t)`` for ``v \in V`` | ``1`` |


The separation problem is equivalent to finding a non-empty minimum-capacity ``s``-``t`` cut ``Y`` in ``\mathcal{D}``. This can be done with the following MILP:

```math
\begin{array}{rll}
	\min \, & \sum\limits_{a \in \mathcal{A}} u_a \beta_a \\
	\mathrm{s.t.} \, & \alpha_s - \alpha_t \geq 1 \\
	& \beta_a \geq \alpha_u - \alpha_v & \text{ for all } a= (u,v) \in \mathcal{A} \\
	& \sum\limits_{v \in V} \alpha_v \geq 1 \\
	& \alpha, \beta \in \{0,1\} 
\end{array}
```
"""

# ╔═╡ 9f0431b8-7f5e-4081-bb09-c8c3014e035b
TODO(md"Implement this better formulation in the `cut_separation_pb` function.

This function must have three outputs in this order
- A boolean `found` telling if a constraint is violated (i.e. if a cut should be added)
- A BitVector representing the edges in set ``Y`` (`Y[e] == 1` if ``e \in E(Y)``)
- The number of vertices in ``Y``.
")

# ╔═╡ 93e86d7a-6045-4f9b-b81a-4c397663fbcb
function build_flow_graph(graph, weights; infinity=1e6)
	V = nv(graph)
	E = ne(graph)

	missing
	return sources, destinations, costs
end

# ╔═╡ 7c1b96bc-b493-4b47-baef-22c6629b8286
function cut_separation_problem(graph, weights; MILP_solver=GLPK.Optimizer, tol=1e-6)
	sources, destinations, costs = build_flow_graph(graph, weights)
	missing
end

# ╔═╡ 60f15817-f696-4fa4-a69d-46c00f2513c7
md"""## 3. Cut generation"""

# ╔═╡ ec711cc2-1613-42d7-bdae-01460509da24
TODO(md"Complete the `cut_generation` function by writing the `my_callback_function`. The `separate_constraint_function` input is expected to either be `cut_separation_pb` or `MILP_separation_pb` defined above.
")

# ╔═╡ 4b952721-31ba-4583-9034-3a5aaec07934
tip(md"See how to use callbacks [here](https://jump.dev/JuMP.jl/stable/manual/callbacks/#callbacks_manual).")

# ╔═╡ 154b4c51-c471-490e-8265-230f3eda92e4
function cut_generation(
    instance::Instance;
	separation_problem=MILP_separation_pb,
    MILP_solver=GLPK.Optimizer,
	verbose=true
)
	# Unpack fields
	(; graph, first_stage_costs, second_stage_costs) = instance
	S = nb_scenarios(instance)
	E = ne(graph)
	V = nv(graph)

	# Initialize model and link to solver
	model = Model(MILP_solver)

	missing
end

# ╔═╡ a129d5aa-1d45-407a-aeb2-00845330a0cb
md"""## 4. Testing"""

# ╔═╡ 0a6fc7ae-acb4-48ef-93ac-02f9ada0fcae
md"Now, we can apply the branch-and-cut on a small instance. However, it will struggle on larger ones because it's quite slow."

# ╔═╡ a01ba16a-1aa7-4bcf-8335-ba67815bfe87
cut_solution = cut_generation(
	small_instance; separation_problem=MILP_separation_problem
)

# ╔═╡ 1528b9b3-2857-41db-8bfa-a4a38e7f71e0
plot_scenario(cut_solution, small_instance, 1)

# ╔═╡ c2ad1f7e-2f4a-46a3-9cbf-852d8a414af2
# cut_sol = cut_generation(
# 	instance; separation_problem=MILP_separation_problem,
# )

# ╔═╡ 85fd6b13-421a-4834-b227-55bba9f12f24
#scenario_slider

# ╔═╡ 2c2b132c-6b66-4d7b-ad43-a812e0d69573
md"""The min-cut formulation being faster, we can apply to a larger instance:"""

# ╔═╡ 104d1d6a-e6ed-4511-b08a-a72315959390
cut_solution_2 = cut_generation(
	instance; separation_problem=cut_separation_problem,
)

# ╔═╡ 5b4cda6b-67e9-4c8a-8e7e-c6dd791f8726
scenario_slider

# ╔═╡ 93d718df-351e-4111-99b5-b7ddaf657955
plot_scenario(cut_solution_2, instance)

# ╔═╡ 0709ba9a-de5a-4e33-88f1-c10a49bfc065
solution_value(cut_solution_2, instance)

# ╔═╡ 7198a8f1-0d37-430a-8c55-d69b3e137cca
is_feasible(cut_solution_2, instance)

# ╔═╡ c10b27d7-c222-47fb-bbb2-3e55cc030e50
md"# III - Column generation"

# ╔═╡ 27cf7e8b-3b5e-4401-a246-3a8949829764
md"""
Since minimum spanning tree can be solved efficiently, it is natural to perform and Dantzig-Wolfe reformulation of the problem previously introduced.

It leads to the following formulation.

```math
    \begin{array}{rll}
        \min\,& \displaystyle\sum_{e \in E}c_e y_e +  \frac{1}{|S|}\sum_{e \in E}\sum_{s \in S}d_{es}z_{es}\\
        \mathrm{s.t.} \,& y_e + z_{es} = \displaystyle\sum_{T \in \mathcal{T}\colon e \in T} \lambda_{T}^s & \text{for all $e\in E$ and $s \in S$} \\
        & \displaystyle\sum_{T \in \mathcal{T}} \lambda_{T}^s = 1 & \text{for all }s \in S \\
        & y,z,\lambda\in \{0,1\}
    \end{array}
```

The linear relaxation of this problem can be solved by column generation, and the problem itself can be solved using a Branch-and-Price. 
"""

# ╔═╡ 0800e2f6-4085-42dc-982c-e2b833b4171a
md"""
The column generation can be implemented as a cut generation in the dual:

```math
\begin{aligned}
	\max_{\nu, \mu}\quad & \sum_{s\in S} \nu_s &\\
	\text{s.t. }\quad & \frac{d_{es}}{|S|} \geq \mu_{es}, & \forall e\in E,\, \forall s\in S\\
	& c_e \geq \sum_{s\in S} \mu_{es}, & \forall e\in E\\
	& \nu_s \leq \sum_{e\in T} \mu_{es}, & \forall s\in S,\, \forall T\in\mathcal{T}\\
	& \mu,\nu\in\mathbb{R}
\end{aligned}
```
"""

# ╔═╡ 8a2baef9-9ab3-4702-a3c2-af8300c83f7d
TODO(md"Implement the `column_generation` using this cut generation formulation.

The function should have four outputs in this order:
- Objective value of the model
- Value of variable `ν`
- Value of variable `μ`
- Vector of all columns added during callbacks
")

# ╔═╡ d3d684c2-28e7-4aa7-b45e-0ccc3247e5d4
function column_generation(instance; MILP_solver=GLPK.Optimizer, tol=1e-6, verbose=true)
	missing
end

# ╔═╡ 6d832dba-3f72-4782-919f-e1c1a0a92d3b
(; ν, μ, columns) = column_generation(instance)

# ╔═╡ 5f815086-5bf4-4a4c-84c2-94f2344cd6dd
md"From this solution of the linear relaxation, we can reconstruct an integer heuristic solution by solving the column formulation and restricting the number of columns."

# ╔═╡ 6d2087d2-ac24-40ac-aadb-fa71dbec6f0e
function column_heuristic(instance, columns; MILP_solver=GLPK.Optimizer)
	(; graph, first_stage_costs, second_stage_costs) = instance
	E = ne(graph)
	S = nb_scenarios(instance)
	T = length(columns)

	model = Model(MILP_solver)

	@variable(model, y[e in 1:E], Bin)
	@variable(model, z[e in 1:E, s in 1:S], Bin)

	@variable(model, λ[t in 1:T, s in 1:S], Bin)

	@objective(
		model, Min,
		sum(first_stage_costs[e] * y[e] for e in 1:E) + sum(second_stage_costs[e, s] * z[e, s] for e in 1:E for s in 1:S) / S
	)

	@constraint(model, [s in 1:S], sum(λ[t, s] for t in 1:T) == 1)
	@constraint(
		model, [e in 1:E, s in 1:S],
		y[e] + z[e, s] == sum(λ[t, s] for t in 1:T if columns[t][e])
	)

	optimize!(model)

	return Solution(value.(y) .> 0.5, value.(z) .> 0.5)
end

# ╔═╡ 59bf6b13-0579-4f95-8acf-3c23d9bb9463
columns

# ╔═╡ 9e23f72a-0220-49d0-8f17-948cce8addbb
column_solution = column_heuristic(instance, columns)

# ╔═╡ 54aa7e04-897d-42d4-9ff9-62d8992397ec
scenario_slider

# ╔═╡ 3f94c697-bb57-4414-babe-74860ec0ac60
plot_scenario(column_solution, instance)

# ╔═╡ 4dcf3e69-0e4f-487f-ae24-b0fac8353908
is_feasible(column_solution, instance)

# ╔═╡ 6a3f4ea6-e780-49c3-b1a9-437aefc404be
solution_value(column_solution, instance)

# ╔═╡ 796be5ea-944b-4827-bfe6-654664c35fb3
md"""# IV - Benders decomposition"""

# ╔═╡ fba7d164-ff1c-4587-92e7-e7fd0668c0bd
md"The integer optimal solution of the column generation formulation can be found using a Branch-and-price, quite heavy to implement. Another option is to apply a Benders decomposition to decouple the scenarios."

# ╔═╡ 1f77d228-4106-4cc8-a9b3-05855f94660e
md"""
When first stage variables ``y`` are fixed, the subproblem for scenario ``s``becomes:

```math
\begin{aligned}
\min_{z, \lambda}\quad & \sum_{e\in E}d_{es} z_{es}\\
\text{s.t.}\quad & z_{es} = \sum_{T\in \mathcal{T}\colon e\in T}\lambda_T^s - y_e & \forall e \in E\\
& \sum_{T\in\mathcal{T}}\lambda_T^s = 1\\
& z, \lambda\geq 0
\end{aligned}
```

We can simplify further the formulation by removing variable ``z``:

```math
\begin{aligned}
\min_{z, \lambda}\quad & \sum_{T\in\mathcal{T}}\sum_{e\in T}d_{es}\lambda_T^s - cst(y_e) \\
\text{s.t.}\quad & \sum_{T\in \mathcal{T}\colon e\in T}\lambda_T^s \geq y_e & \forall e \in E\\
& \sum_{T\in\mathcal{T}}\lambda_T^s = 1\\
& z, \lambda\geq 0
\end{aligned}
```

We take its dual:
```math
\begin{aligned}
\max_{\mu, \nu}\quad & \nu_s + \sum_{e\in E} y_e \mu_{es} - cst \\
\text{s.t.}\quad & \sum_{e\in T} (d_{es} - \mu_{es}) - \nu_s \geq 0, & \forall T\in\mathcal{T}\\
& \mu\geq 0, \nu\in\mathbb{R}
\end{aligned}
```

This dual can be solved using constraint generation, with once again a usual minimum spanning tree separation problem that can be solved using Kruskal algorithm:
```math
\min \sum_{e\in T} (d_{es} - \mu_{es})
```

If the primal is feasible, we generate an optimality cut:
```math
\theta_s \geq \nu_s + \sum_{e\in E} \mu_{es}y_e - \sum_{e\in E} d_{es} y_e
```

When the primal is unfeasible, there is an unbounded ray for the dual, i.e. ``\mu, \nu`` such that ``\nu_s + \sum_e \mu_{es} y_e > 0`` and ``-\nu_s - \sum_{e\in T}\mu_{es}``. (``\alpha \nu`` and ``\alpha\mu`` are also solutions for all ``\alpha > 0``). Such solution can be found by solving:

```math
\begin{aligned}
\max_{\mu, \nu}\quad & \nu_s + \sum_{e\in E} \mu_{es} y_e \\
\text{s.t.}\quad & -\nu_s - \sum_{e\in T}\mu_{es} \geq 0 & \forall T\in \mathcal{T}\\
& 0 \leq \mu_{es} \leq 1 & \forall e\in E\\
& \nu_s\leq 1
\end{aligned}
```

Let us denote ``\mathcal{F}`` the feasibility cuts and ``\mathcal{O}_s`` the optimality cuts set. We obtain the following Benders master problem:

```math
\begin{aligned}
\max_{y}\quad & \sum_{e\in E} c_e y_e + \frac{1}{|S|}\sum_{s\in S}\theta_s \\
\text{s.t.}\quad & \theta_s \geq \nu_s + \sum_{e\in E} \mu_{es} y_e - \sum_{e\in E} d_{es} y_e & \forall s\in S,\, \forall (\nu, \mu) \in \mathcal{O}_s\\
& \nu + \sum_{e\in E} \mu_e y_e & \forall (\nu, \mu)\in \mathcal{F}\\
& y\in\{0, 1\}
\end{aligned}
```
"""

# ╔═╡ 5af768d8-c1f4-4721-bf14-bd43517e609c
TODO("Implement the benders decomposition described above")

# ╔═╡ b14ce83b-ac6a-4baf-b499-16dedff13fa3
function separate_benders_cut(instance::Instance, y, s; MILP_solver, tol=1e-5)
	missing
end;

# ╔═╡ 2c7ac71d-2983-4da2-8ebf-748e51bd9d08
function benders_decomposition(
    instance::Instance;
    MILP_solver=GLPK.Optimizer,
	tol=1e-6,
	verbose=true
)
	missing
end

# ╔═╡ fd3a09c5-6b02-4382-9eaf-fa81e4589057
benders_solution = benders_decomposition(instance)

# ╔═╡ 4c928cc0-71dc-4bd3-9486-0b1bfb7220d5
scenario_slider

# ╔═╡ 404b7809-6718-424f-8aec-a8b2c35701eb
plot_scenario(benders_solution, instance)

# ╔═╡ bab8cb56-bf9a-4a72-a3f9-98bba93a18ef
is_feasible(benders_solution, instance)

# ╔═╡ 9a0d2a0d-1e07-4f9f-ab5d-2b8dcb522163
solution_value(benders_solution, instance)

# ╔═╡ 74fc70cc-42e7-4e20-9c04-cebe2dcbd3f3
md"# V - Lagrangian Relaxation"

# ╔═╡ 78ec76dd-db39-4ad0-8db5-559839420d96
md"""
### 1. Lagrangian relaxation formulation

Let us introduce one copy of first stage variables ``y`` per scenario. An equivalent formulation of the problem is

```math
\begin{array}{ll}
\min\, & \displaystyle \sum_{e\in E}c_e y_e + \sum_{e \in E} \sum_{s \in S}d_{es}z_{es} \\
\mathrm{s.t.}\, & \mathbf{y}_s + \mathbf{z}_s \in \mathcal{P}, \quad\quad \text{for all $s$ in $S$}  \\
& y_{es} = y_e, \quad \quad \quad \,\text{for all $e$ in $E$ and $s$ in $S$}
\end{array}
```

Let us relax (dualize) the constraint ``y_{es} = y_e``. We denote by ``\theta_{es}`` the associated Lagrange multiplier.

The Lagrangian dual problem becomes

```math
\begin{array}{rlrlrl}
\max_{\theta}\mathcal{G}(\theta)= \min_{y}& \sum_{e \in E}(c_e + \frac{1}{|S|}\sum_{s \in S} \theta_{es})y_e \\
&+ \frac{1}{|S|}\sum_{s \in S}\min_{\mathbf{y}_s,\mathbf{z}_s} \sum_{e \in E}d_{es}z_{es} - \theta_{es}y_{es}\\
\mathrm{s.t.} & 0 \leq \mathbf{y} \leq M\\
& \mathbf{y}_s + \mathbf{z}_s \in \mathcal{P}, \quad\quad \text{for all $s$ in $S$}  
\end{array}
```

where ``M`` is a large constant. 
In theory, we would take ``M=+\infty``, but taking a finite ``M`` leads to more informative gradients.
"""

# ╔═╡ aee74231-afe1-4793-b12a-89948473b6fb
md"""Solving the first stage subproblem amounts to checking the sign of ``c_e + \frac{1}{|S|}\sum_{s \in S} \theta_{es}``:"""

# ╔═╡ 2ca79687-187a-4259-9470-65d59e537749
function first_stage_optimal_solution(inst::Instance, θ::AbstractMatrix; M=20.0)
	S = nb_scenarios(inst)
	E = ne(inst.graph)

	# first stage objective value
    edge_weight_vector = inst.first_stage_costs .+ vec(sum(θ; dims=2)) ./ S

    edges_index_with_negative_cost = [e for e in 1:E if edge_weight_vector[e] < 0]

    value = 0.0
    if length(edges_index_with_negative_cost) > 0
        value = sum(M * edge_weight_vector[e] for e in edges_index_with_negative_cost)
    end

    grad = zeros(E, S)
	grad[edges_index_with_negative_cost, :] .= M / S
    return value, grad
end;

# ╔═╡ da57e0b6-e917-4d86-87d4-07aeac0bbdb2
md"""The optimal solution of the second stage problem can be computed using Kruskal's algorithm:"""

# ╔═╡ afaf3780-52bc-452b-b966-6b4f59166e66
function second_stage_optimal_solution!(
    instance::Instance,
    θ::AbstractMatrix,
    scenario::Int,
    grad::AbstractMatrix,
)
	(; graph, second_stage_costs) = instance
    S = nb_scenarios(instance)

	weights = min.(-θ[:, scenario], second_stage_costs[:, scenario])

    (; value, tree) = kruskal(graph, weights)

	# update gradient
	slice = (-θ[:, scenario] .< second_stage_costs[:, scenario]) .&& tree
	grad[slice, scenario] .-= 1 / S

    return value ./ S
end;

# ╔═╡ 368cc605-eff3-4f78-b295-68c140a273db
md"""
### 2. Lagrangian dual function and its gradient

We have

```math
(\nabla \mathcal{G}(\theta))_{es}= \frac{1}{|S|} (y_e - y_{es}).
```

Considering the sum on the second stage scenarios as an expectation, we can get stochastic gradients and maximize $\mathcal{G}$ using gradient ascent.
"""

# ╔═╡ a89285de-3f50-44c2-8844-0debbc577ce6
function lagrangian_function_value_gradient(inst::Instance, θ::AbstractMatrix)
    value, grad = first_stage_optimal_solution(inst, θ)

	S = nb_scenarios(inst)
    values = zeros(S)
    for s in 1:S
		# Different part of grad are modified
        values[s] = second_stage_optimal_solution!(inst, θ, s, grad)
    end
    value += sum(values)
    return value, grad
end;

# ╔═╡ 2481e7f5-4db0-4db2-a4db-24512eb5d6df
md"""### 3. Lagrangian heuristic

Once a solution of the relaxed problem is found, we have one solution $y_s$ per scenario $s$. We can then use an heuristic to reconstruct a good first stage decision $y$.

Below is an example implementation of such an heuristic:
"""

# ╔═╡ f6dbbcb6-a16e-4630-ac74-06ce3d12e040
function lagrangian_heuristic(θ::AbstractMatrix; inst::Instance)
    # Retrieve - y_{es} / S from θ by computing the gradient
	(; graph) = inst
	S = nb_scenarios(inst)
    grad = zeros(ne(graph), S)
    for s in 1:S
        second_stage_optimal_solution!(inst, θ, s, grad)
    end
    # Compute the average (over s) y_{es} and build a graph that is a candidate spannning tree (but not necessarily a spanning tree nor a forest)
    average_x = -vec(sum(grad; dims=2))
    weights = average_x .> 0.5
    # Build a spanning tree that contains as many edges of our candidate as possible
    _, tree_from_candidate = kruskal(graph, weights; minimize=false)
    # Keep only the edges that are in the initial candidate graph and in the spanning tree
    forest = weights .&& tree_from_candidate
    sol = solution_from_first_stage_forest(forest, inst)
	# v, _ = evaluate_first_stage_solution(inst, forest)
    return solution_value(sol, inst), forest
end;

# ╔═╡ 7934b593-36c3-4c70-b51b-aaf8f219aaf3
md"""### 4. Main algorithm"""

# ╔═╡ a6186708-a499-43bc-89e2-99a4abd0b700
TODO(md"Implement the full lagrangian relaxation using all the functions defined above.")

# ╔═╡ e6f62957-7cba-48a1-bfd3-52e796c71fed
md"""### 5. Testing"""

# ╔═╡ a75f1918-a7e3-4041-a4bf-b64fb7094546
begin
	f = plot()
	plot!(f, ub_history; label="Upper bound: lagrangian heuristic", color=:orange)
	plot!(f, lb_history; label="Lower bound: lagrangian relaxation", color=:purple)
	f
end

# ╔═╡ fc1b22a9-e3bd-470b-82c2-c3cf3cd62052
scenario_slider

# ╔═╡ 5b4f8555-a6dc-4869-afe6-91820f76155d
plot_scenario(lagrangian_solution, instance)

# ╔═╡ b6c76c74-6931-4bf5-a126-ae2ac3f19607
is_feasible(lagrangian_solution, instance)

# ╔═╡ ac1f93b7-57f0-41ed-b159-800863c6a530
solution_value(lagrangian_solution, instance)

# ╔═╡ d07d3192-32bd-4f8d-8775-23429e888eb6
function lagrangian_relaxation(
    inst::Instance; nb_epochs=100, stop_gap=1e-8
)
	missing
    return solution, (; lb, ub, best_theta, lb_history, ub_history)
end

# ╔═╡ 5529ac14-1171-4e77-b3b4-dbcde4b704a4
lagrangian_solution, (; lb, ub, ub_history, lb_history) = lagrangian_relaxation(instance; nb_epochs=25_000)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoGrader = "5e01e182-cc8e-4f07-9d5c-10e469e58e66"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
DataStructures = "~0.18.15"
Flux = "~0.14.8"
GLPK = "~1.1.3"
Graphs = "~1.9.0"
JuMP = "~1.18.0"
Plots = "~1.39.0"
PlutoGrader = "~0.1.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
ProgressLogging = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "6a09a896b93db6daceed9ba31d334aa3660cd0bd"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "0aa0a3dd7b9bacbbadf1932ccbdfa938985c5561"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.58.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "2118cb2765f8197b08e5958cdd17c165427425ee"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.19.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "c0ae2a86b162fb5d7acc65269b469ff5b8a73594"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.Configurations]]
deps = ["ExproniconLite", "OrderedCollections", "TOML"]
git-tree-sha1 = "4358750bb58a3caefd5f37a4a0c5bfdbbf075252"
uuid = "5218b696-f38b-4ac9-8b61-a12ec717816d"
version = "0.17.6"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "bce17cd0180a75eec637d6e3f8153011b8bdb25a"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.0.0"

[[deps.ExproniconLite]]
git-tree-sha1 = "fbc390c2f896031db5484bc152a7e805ecdfb01f"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.5"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "e3b646440f6d0af12c44402c1b9a73f7dcd1157d"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.8"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.FuzzyCompletions]]
deps = ["REPL"]
git-tree-sha1 = "c8d37d615586bea181063613dccc555499feb298"
uuid = "fb4132e2-a121-4a70-b8a1-d5b831dcdcc2"
version = "0.5.3"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "e37c68890d71c2e6555d3689a5d5fc75b35990ef"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.1.3"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.2.1+6"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "85d7fb51afb3def5dcb85ad31c3707795c8bccc1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "9.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "899050ace26649433ef1af25bc17a815b3db52b7"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.9.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "8aa91235360659ca7560db43a7d57541120aa31d"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.11"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "769d01cf0d3d1f3e59594cc43c8b319b36d7c2a3"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.18.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "e49bce680c109bc86e3e75ebcb15040d6ad9e1d3"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.27"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "653e0824fc9ab55b3beec67a6dbbe514a65fb954"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.15"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "0678579657515e88b6632a3a482d39adcbb80445"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.1"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "98eaee04d96d973e79c25d49167668c5c8fb50e2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "8f7f3cabab0fd1800699663533b6d5cb3fc0e612"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.2.2"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "38756922d32476c8f41f73560b910fc805a5a103"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.4.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "b211c553c199c111d998ecdaf7623d1b89b69f93"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.12"

[[deps.Malt]]
deps = ["Distributed", "Logging", "RelocatableFolders", "Serialization", "Sockets"]
git-tree-sha1 = "18cf4151e390fce29ca846b92b06baf9bc6e002e"
uuid = "36869731-bdee-424d-aa32-cab38c994e3b"
version = "1.1.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "d2a140e551c9ec9028483e3c7d1244f417567ac0"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.24.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "900a11b3a2b02e36b25cb55a80777d4a4670f0f6"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.10"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "5e4029759e8699ec12ebdf8721e51a659443403c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "34205b1204cc83c43cd9cfe53ffbd3b310f6e8c5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.1"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Pluto]]
deps = ["Base64", "Configurations", "Dates", "Downloads", "ExpressionExplorer", "FileWatching", "FuzzyCompletions", "HTTP", "HypertextLiteral", "InteractiveUtils", "Logging", "LoggingExtras", "MIMEs", "Malt", "Markdown", "MsgPack", "Pkg", "PrecompileSignatures", "PrecompileTools", "REPL", "RegistryInstances", "RelocatableFolders", "Scratch", "Sockets", "TOML", "Tables", "URIs", "UUIDs"]
git-tree-sha1 = "e6a92bf27d9e8eda41b672772ad05f6652513e02"
uuid = "c3e4b0f8-55cb-11ea-2926-15256bba5781"
version = "0.19.36"

[[deps.PlutoGrader]]
deps = ["HypertextLiteral", "Markdown", "Pluto", "PlutoTeachingTools", "PlutoUI", "Reexport", "Test"]
git-tree-sha1 = "56d024efcbe50973b9380d44cc871fbfe7389f4f"
repo-rev = "main"
repo-url = "https://github.com/lucaferranti/PlutoGrader.jl"
uuid = "5e01e182-cc8e-4f07-9d5c-10e469e58e66"
version = "0.1.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "89f57f710cc121a7f32473791af3d6beefc59051"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.14"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PrecompileSignatures]]
git-tree-sha1 = "18ef344185f25ee9d51d80e179f8dad33dc48eb1"
uuid = "91cefc8d-f054-46dc-8f8c-26e11d7c5411"
version = "3.0.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "116d71e489abc472efa460cfa2bc0ac7cd0bab54"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "4e17a790909b17f7bf1496e3aec138cf01b60b3b"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.0"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "e579d3c991938fecbb225699e8f611fa3fbf2141"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.79"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "30c1b8bfc2b3c7c5d8bba7cd32e8b6d5f968e7c3"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.68"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "9d749cd449fb448aeca4feee9a2f4186dbb5d184"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.4"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─6cd3608b-0df6-4d22-af6c-9b6cb1d955c3
# ╠═33d91779-378b-4e57-a779-34cf25045d2b
# ╠═e11f28b6-8f91-11ee-088d-d51c110208c6
# ╠═1b51af71-8474-46e6-91b2-35fc9adb2c5a
# ╠═30670737-22c2-42e1-a0a4-43aa0fa70752
# ╟─ab5c0cf3-6ce9-4908-8c09-b664503e5ac1
# ╟─bcc75aa2-9cd7-49bf-ad46-9dd5a8db3ef0
# ╟─4ca8f8f1-058f-4b47-adce-4cdbe916d628
# ╟─06b5c71a-bb44-4694-be43-1b3e9b38ece2
# ╟─2dd444e9-5df7-43c0-953c-b705bfc024a3
# ╟─aee968cd-1d4f-40e4-8e61-14a92bb89989
# ╠═3f89b439-03e7-4e1e-89ab-63bbf5fa2194
# ╠═0f4090e3-864c-46e5-bb28-203e735c63a8
# ╟─4eab9c97-9278-4895-ba2d-1ddb78afe530
# ╠═5e867265-c733-485a-b39a-c4320e99c92a
# ╟─21f02f67-35a2-4ff0-9343-58562d5e5bfb
# ╟─da2b7fef-627f-4b4a-83dc-0e731a243c61
# ╠═8249ad29-f900-4992-9c32-60860d2973ee
# ╟─b4c0b7f5-8863-4921-915f-c7b73cb1e792
# ╠═b9daab11-d807-40fd-b94b-bc79ae80275e
# ╟─77630435-3536-4714-b4c7-db4473e7ba0e
# ╠═2251158b-d21a-4e4e-bc11-89bf7c385557
# ╠═55ca5072-6831-4794-8aed-68d8b56f7f80
# ╟─2f9c2388-a178-452b-a013-a2cc1cabc4b4
# ╟─a8e889d5-a7bc-4c2e-9383-6f156eb2dd6a
# ╠═de304db1-e5ca-4aaa-9ea7-d271bec8ae7d
# ╟─eea94e99-cc33-4464-ac24-587466b17e48
# ╟─c6cd42d1-c428-49a7-99a4-93f342373f06
# ╟─ad4284c3-a926-4c6b-8c32-4d24bcbede60
# ╟─bf369999-41c1-481f-9f17-ec7d5dd08445
# ╟─7d9a2b6e-e8a9-4cf0-af4b-e45603d45008
# ╠═a3eeb63a-b971-4806-9146-74936d4cc2e6
# ╠═e14e5513-5cc2-4b70-ab29-8ee53ca166cc
# ╠═c34c3f25-58ea-4219-b856-2ed9d790d291
# ╠═b0155649-8f26-47ac-9d80-95a979f716cb
# ╠═c111dadd-3cb6-4cb0-b082-b67e11248e1c
# ╠═8bc212ec-5a5d-401d-97d0-b2e0eb2b3b6f
# ╠═8c00c839-b349-42e1-8e3f-afbd74fcf8c2
# ╠═d646e96c-5b2c-4349-bf11-133494af1453
# ╠═9d2b37d1-8a73-4b3e-853a-d849b7895d01
# ╠═53a4d6de-b798-4773-830f-a26d56241b1e
# ╠═f81105f1-a70e-406c-ad7e-0390910e4c17
# ╟─76cbf0da-7437-464a-ba1b-e093cabd3b83
# ╠═71ad5432-3c86-43da-b097-c668388b836b
# ╠═6186efdf-227e-4e95-b788-5dd3219162e7
# ╟─5d2f732b-2903-45f1-aa27-4c0df5e8645b
# ╟─49df95f6-34b8-48d1-b1de-40309b27c48a
# ╟─bbedc946-cf38-4ba9-b631-0d00e5807f01
# ╟─b104ec32-2b7a-42f2-99ee-6dee7c0c9cad
# ╟─02f6f906-c744-4e16-b73e-2dc098d6d7e3
# ╟─16c03d54-720e-493b-bde6-34d4da9941ab
# ╠═d9441eda-d807-4452-af10-11804bc668da
# ╟─3eaae7fd-8fff-43b7-9945-cdbde3b6c0fe
# ╟─7f8dea10-942e-4bae-9223-387650e35cc9
# ╟─9f0431b8-7f5e-4081-bb09-c8c3014e035b
# ╠═93e86d7a-6045-4f9b-b81a-4c397663fbcb
# ╠═7c1b96bc-b493-4b47-baef-22c6629b8286
# ╟─60f15817-f696-4fa4-a69d-46c00f2513c7
# ╟─ec711cc2-1613-42d7-bdae-01460509da24
# ╟─4b952721-31ba-4583-9034-3a5aaec07934
# ╠═154b4c51-c471-490e-8265-230f3eda92e4
# ╟─a129d5aa-1d45-407a-aeb2-00845330a0cb
# ╟─0a6fc7ae-acb4-48ef-93ac-02f9ada0fcae
# ╠═a01ba16a-1aa7-4bcf-8335-ba67815bfe87
# ╠═1528b9b3-2857-41db-8bfa-a4a38e7f71e0
# ╠═c2ad1f7e-2f4a-46a3-9cbf-852d8a414af2
# ╠═85fd6b13-421a-4834-b227-55bba9f12f24
# ╟─2c2b132c-6b66-4d7b-ad43-a812e0d69573
# ╠═104d1d6a-e6ed-4511-b08a-a72315959390
# ╟─5b4cda6b-67e9-4c8a-8e7e-c6dd791f8726
# ╠═93d718df-351e-4111-99b5-b7ddaf657955
# ╠═0709ba9a-de5a-4e33-88f1-c10a49bfc065
# ╠═7198a8f1-0d37-430a-8c55-d69b3e137cca
# ╟─c10b27d7-c222-47fb-bbb2-3e55cc030e50
# ╟─27cf7e8b-3b5e-4401-a246-3a8949829764
# ╟─0800e2f6-4085-42dc-982c-e2b833b4171a
# ╟─8a2baef9-9ab3-4702-a3c2-af8300c83f7d
# ╠═d3d684c2-28e7-4aa7-b45e-0ccc3247e5d4
# ╠═6d832dba-3f72-4782-919f-e1c1a0a92d3b
# ╟─5f815086-5bf4-4a4c-84c2-94f2344cd6dd
# ╠═6d2087d2-ac24-40ac-aadb-fa71dbec6f0e
# ╠═59bf6b13-0579-4f95-8acf-3c23d9bb9463
# ╠═9e23f72a-0220-49d0-8f17-948cce8addbb
# ╟─54aa7e04-897d-42d4-9ff9-62d8992397ec
# ╠═3f94c697-bb57-4414-babe-74860ec0ac60
# ╠═4dcf3e69-0e4f-487f-ae24-b0fac8353908
# ╠═6a3f4ea6-e780-49c3-b1a9-437aefc404be
# ╟─796be5ea-944b-4827-bfe6-654664c35fb3
# ╟─fba7d164-ff1c-4587-92e7-e7fd0668c0bd
# ╟─1f77d228-4106-4cc8-a9b3-05855f94660e
# ╟─5af768d8-c1f4-4721-bf14-bd43517e609c
# ╠═b14ce83b-ac6a-4baf-b499-16dedff13fa3
# ╠═2c7ac71d-2983-4da2-8ebf-748e51bd9d08
# ╠═fd3a09c5-6b02-4382-9eaf-fa81e4589057
# ╟─4c928cc0-71dc-4bd3-9486-0b1bfb7220d5
# ╠═404b7809-6718-424f-8aec-a8b2c35701eb
# ╠═bab8cb56-bf9a-4a72-a3f9-98bba93a18ef
# ╠═9a0d2a0d-1e07-4f9f-ab5d-2b8dcb522163
# ╟─74fc70cc-42e7-4e20-9c04-cebe2dcbd3f3
# ╟─78ec76dd-db39-4ad0-8db5-559839420d96
# ╟─aee74231-afe1-4793-b12a-89948473b6fb
# ╠═2ca79687-187a-4259-9470-65d59e537749
# ╟─da57e0b6-e917-4d86-87d4-07aeac0bbdb2
# ╠═afaf3780-52bc-452b-b966-6b4f59166e66
# ╟─368cc605-eff3-4f78-b295-68c140a273db
# ╠═a89285de-3f50-44c2-8844-0debbc577ce6
# ╟─2481e7f5-4db0-4db2-a4db-24512eb5d6df
# ╠═f6dbbcb6-a16e-4630-ac74-06ce3d12e040
# ╟─7934b593-36c3-4c70-b51b-aaf8f219aaf3
# ╟─a6186708-a499-43bc-89e2-99a4abd0b700
# ╠═d07d3192-32bd-4f8d-8775-23429e888eb6
# ╟─e6f62957-7cba-48a1-bfd3-52e796c71fed
# ╠═5529ac14-1171-4e77-b3b4-dbcde4b704a4
# ╠═a75f1918-a7e3-4041-a4bf-b64fb7094546
# ╟─fc1b22a9-e3bd-470b-82c2-c3cf3cd62052
# ╠═5b4f8555-a6dc-4869-afe6-91820f76155d
# ╠═b6c76c74-6931-4bf5-a126-ae2ac3f19607
# ╠═ac1f93b7-57f0-41ed-b159-800863c6a530
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
