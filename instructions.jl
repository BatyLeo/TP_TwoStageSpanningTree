### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ caa8d157-a371-4e66-8d4c-d027ec9e20e2
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
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
easy_instance = random_instance(; n, m, nb_scenarios=1, seed=0)

# ╔═╡ c541b1a0-553c-4f91-80c9-e995d6b13039
easy_value = kruskal(easy_instance.graph, min.(easy_instance.first_stage_costs, easy_instance.second_stage_costs[:, 1])).value

# ╔═╡ c111dadd-3cb6-4cb0-b082-b67e11248e1c
S = 20

# ╔═╡ 8bc212ec-5a5d-401d-97d0-b2e0eb2b3b6f
instance = random_instance(; n, m, nb_scenarios=S, seed=0)

# ╔═╡ 8c00c839-b349-42e1-8e3f-afbd74fcf8c2
@kwdef struct Solution
	y::BitVector
	z::BitMatrix
end

# ╔═╡ d646e96c-5b2c-4349-bf11-133494af1453
# check if given input is a spanning tree
function is_spanning_tree(tree_candidate::BitVector, graph::AbstractGraph)
    edge_list = [e for (i, e) in enumerate(edges(graph)) if tree_candidate[i]]
    subgraph = induced_subgraph(graph, edge_list)[1]
    return !is_cyclic(subgraph) && nv(subgraph) == nv(graph)
end

# ╔═╡ 9d2b37d1-8a73-4b3e-853a-d849b7895d01
# Check if given input solution is feasible for instance
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
# Retrieve a full solution from given first stage forest solution
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
# Compute the objective value of solution for instance
function solution_value(solution::Solution, instance::Instance)
    return dot(solution.y, instance.first_stage_costs) + dot(solution.z, instance.second_stage_costs) / nb_scenarios(instance)
end

# ╔═╡ 76cbf0da-7437-464a-ba1b-e093cabd3b83
md"""### Visualization tools"""

# ╔═╡ 6186efdf-227e-4e95-b788-5dd3219162e7
begin
	scenario_slider = @bind current_scenario PlutoUI.Slider(1:S; default=1, show_value=true);
end;

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

Finding the most violated constraint (for a given scenario $s$) is called the **separation problem**, and can be formulated as:

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
md"""The separation problem can be formulated as the following MILP (why?):

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

# ╔═╡ 8c6610f7-5581-42c3-9792-d7c604e58b2c
hint(md"""
- `found` = ``\sum\limits_{v\in V}\alpha_v^\star - 1 - \sum\limits_{e \in E} \beta_e^\star (y_e + z_{es}) < 0``
- Second output is ``\beta^\star`` binary value
- ``|Y| = \sum\limits_{v\in V} \alpha_v^\star``
""")

# ╔═╡ d9441eda-d807-4452-af10-11804bc668da
"""
	MILP_separation_problem(graph, weights; MILP_solver, tol=1e-6)

# Arguments
- `graph`: graph instance of the separation problem
- `weights`: vector indexed by edge indices, weights[e] = ``y_e + z_{es}``

# Keyword arguments
- `MILP_solver`: MIP solver to use
- `tol`: tolerance to avoid numerical issues
"""
function MILP_separation_problem(graph, weights; MILP_solver, tol=1e-6)
	V = nv(graph)
	E = ne(graph)

	model = Model(MILP_solver)
	set_silent(model)
	missing
end

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

# ╔═╡ 1cb00839-476b-453b-bad8-2a65b159819b
md"The digraph ``\mathcal{D}`` can be built using the following function:"

# ╔═╡ 93e86d7a-6045-4f9b-b81a-4c397663fbcb
"""
	build_flow_graph(graph, weights; infinity=1e6)

Build the underlying flow graph from initial graph and weights ``y_e + z_{es}``

# Outputs
Three vectors indexed by arc indices `a` of the digraph
- `sources`: sources[a] = source vertex index of arc `a`
- `destinations`: destinations[a] = destination vertex of arc `a`
- `costs`: costs[a] = capacity of arc `a`
"""
function build_flow_graph(graph, weights; infinity=1e6)
	V = nv(graph)
	E = ne(graph)

	# A = 3 * E + V
	VV = 2 + E + V

	o = 1
	d = 2

	sources = vcat(
		fill(o, E),
		[2 + e for e in 1:E],
		[2 + e for e in 1:E],
		[2 + E + v for v in 1:V]
	)
	destinations = vcat(
		[2 + e for e in 1:E],
		[2 + E + src(e) for e in edges(graph)],
		[2 + E + dst(e) for e in edges(graph)],
		fill(d, V)
	)
	costs = vcat(
		[weights[e] for e in 1:E],
		fill(infinity, 2 * E),
		ones(V)
	)

	return sources, destinations, costs
end;

# ╔═╡ 9f0431b8-7f5e-4081-bb09-c8c3014e035b
TODO(md"Implement this better formulation in the `cut_separation_pb` function.

This function must have three outputs in this order
- A boolean `found` telling if a constraint is violated (i.e. if a cut should be added)
- A BitVector representing the edges in set ``Y`` (`Y[e] == 1` if ``e \in E(Y)``)
- The number of vertices in ``Y``.
")

# ╔═╡ 7c1b96bc-b493-4b47-baef-22c6629b8286
function cut_separation_problem(graph, weights; MILP_solver=GLPK.Optimizer, tol=1e-6)
	sources, destinations, costs = build_flow_graph(graph, weights)
	missing
end

# ╔═╡ 60f15817-f696-4fa4-a69d-46c00f2513c7
md"""## 4. Cut generation"""

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
md"""## 5. Testing"""

# ╔═╡ 0a6fc7ae-acb4-48ef-93ac-02f9ada0fcae
md"Now, we can apply the branch-and-cut on a small instance. However, it will struggle on larger ones because it's quite slow."

# ╔═╡ c2ad1f7e-2f4a-46a3-9cbf-852d8a414af2
easy_cut_solution = cut_generation(
	easy_instance; separation_problem=MILP_separation_problem,
)

# ╔═╡ fdd03643-63d4-4836-8f34-3259f7574fec
solution_value(easy_cut_solution, easy_instance)

# ╔═╡ 79196107-93ac-4eb9-bdf3-87b38cedda38
plot_scenario(easy_cut_solution, easy_instance, 1)

# ╔═╡ 2c2b132c-6b66-4d7b-ad43-a812e0d69573
md"""The min-cut formulation being faster, we can also apply it to a larger instance:"""

# ╔═╡ 53b80d45-5eaa-4be8-a729-ee5e43477885
easy_cut_solution_2 = cut_generation(
	easy_instance; separation_problem=cut_separation_problem,
)

# ╔═╡ 104d1d6a-e6ed-4511-b08a-a72315959390
cut_solution = cut_generation(
	instance; separation_problem=cut_separation_problem,
)

# ╔═╡ 0709ba9a-de5a-4e33-88f1-c10a49bfc065
solution_value(cut_solution, instance)

# ╔═╡ 5b4cda6b-67e9-4c8a-8e7e-c6dd791f8726
scenario_slider

# ╔═╡ 93d718df-351e-4111-99b5-b7ddaf657955
plot_scenario(cut_solution, instance)

# ╔═╡ 96bb6208-c718-48e1-80d2-0e3f9fcc1127
@testset ExerciseScore begin
	@test is_feasible(easy_cut_solution, easy_instance)
	@test is_feasible(cut_solution, instance)
	@test solution_value(easy_cut_solution, easy_instance) == solution_value(easy_cut_solution_2, easy_instance)
	easy_sol = cut_generation(
		easy_instance; separation_problem=cut_separation_problem,
	)
	@test solution_value(easy_sol, easy_instance) == easy_value
end

# ╔═╡ c10b27d7-c222-47fb-bbb2-3e55cc030e50
md"# III - Column generation"

# ╔═╡ 27cf7e8b-3b5e-4401-a246-3a8949829764
md"""
Since minimum spanning tree can be solved efficiently, it is natural to perform a Dantzig-Wolfe reformulation of the problem previously introduced.
We denote by $\mathcal{T}$ the set of spanning trees over tyhe graph.

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
The column generation can be easily implemented as a cut generation in the dual:

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

# ╔═╡ 6ee5026a-387b-4b30-acb7-303cb9da8724
warning_box(md"In order for a callback to be called, the underlying optimization model needs to contain integer variables. In the case of the column generation, all variables are continuous, therefore you need to artificially add a dummy integer variable to the model: `@variable(model, dummy, Bin)`")

# ╔═╡ d3d684c2-28e7-4aa7-b45e-0ccc3247e5d4
function column_generation(
	instance; MILP_solver=GLPK.Optimizer, tol=1e-6, verbose=true
)
	missing
end;

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

# ╔═╡ 2ba1d475-e39b-4bee-8597-c468677a976d
@testset ExerciseScore begin
	@test is_feasible(column_solution, instance)
	@test solution_value(column_solution, instance) >= solution_value(cut_solution, instance)
end

# ╔═╡ 796be5ea-944b-4827-bfe6-654664c35fb3
md"""# IV - Benders decomposition"""

# ╔═╡ fba7d164-ff1c-4587-92e7-e7fd0668c0bd
md"The integer optimal solution of the column generation formulation can be found using a Branch-and-price, quite heavy to implement. Another option is to apply a Benders decomposition to decouple the scenarios."

# ╔═╡ 1f77d228-4106-4cc8-a9b3-05855f94660e
md"""
When first stage variables ``y`` are fixed, the subproblem for scenario ``s`` becomes:

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
\min_{\lambda}\quad & \sum_{T\in\mathcal{T}}\sum_{e\in T}d_{es}\lambda_T^s - constant(y_e) \\
\text{s.t.}\quad & \sum_{T\in \mathcal{T}\colon e\in T}\lambda_T^s \geq y_e & \forall e \in E\\
& \sum_{T\in\mathcal{T}}\lambda_T^s = 1\\
& \lambda\geq 0
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

# ╔═╡ 17d6b9cf-1557-4b13-82cd-e642219ba8ac
@testset ExerciseScore begin
	@test is_feasible(benders_solution, instance)
	@test solution_value(benders_solution, instance) == solution_value(cut_solution, instance)
	easy_sol = benders_decomposition(easy_instance; verbose=false)
	@test solution_value(easy_sol, easy_instance) == easy_value
end

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

# ╔═╡ c6ea482d-431b-41c8-8478-3b795c59c18f
@testset ExerciseScore begin
	@test is_feasible(lagrangian_solution, instance)
	@test solution_value(lagrangian_solution, instance) >= solution_value(cut_solution, instance)
	@test lb <= ub
end

# ╔═╡ d07d3192-32bd-4f8d-8775-23429e888eb6
function lagrangian_relaxation(
    inst::Instance; nb_epochs=100, stop_gap=1e-8
)
	missing
    return solution, (; lb, ub, best_theta, lb_history, ub_history)
end

# ╔═╡ 5529ac14-1171-4e77-b3b4-dbcde4b704a4
lagrangian_solution, (; lb, ub, ub_history, lb_history) = lagrangian_relaxation(instance; nb_epochs=30_000)

# ╔═╡ Cell order:
# ╟─6cd3608b-0df6-4d22-af6c-9b6cb1d955c3
# ╠═caa8d157-a371-4e66-8d4c-d027ec9e20e2
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
# ╠═c541b1a0-553c-4f91-80c9-e995d6b13039
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
# ╟─8c6610f7-5581-42c3-9792-d7c604e58b2c
# ╠═d9441eda-d807-4452-af10-11804bc668da
# ╟─3eaae7fd-8fff-43b7-9945-cdbde3b6c0fe
# ╟─7f8dea10-942e-4bae-9223-387650e35cc9
# ╟─1cb00839-476b-453b-bad8-2a65b159819b
# ╠═93e86d7a-6045-4f9b-b81a-4c397663fbcb
# ╟─9f0431b8-7f5e-4081-bb09-c8c3014e035b
# ╠═7c1b96bc-b493-4b47-baef-22c6629b8286
# ╟─60f15817-f696-4fa4-a69d-46c00f2513c7
# ╟─ec711cc2-1613-42d7-bdae-01460509da24
# ╟─4b952721-31ba-4583-9034-3a5aaec07934
# ╠═154b4c51-c471-490e-8265-230f3eda92e4
# ╟─a129d5aa-1d45-407a-aeb2-00845330a0cb
# ╟─0a6fc7ae-acb4-48ef-93ac-02f9ada0fcae
# ╠═c2ad1f7e-2f4a-46a3-9cbf-852d8a414af2
# ╠═fdd03643-63d4-4836-8f34-3259f7574fec
# ╠═79196107-93ac-4eb9-bdf3-87b38cedda38
# ╟─2c2b132c-6b66-4d7b-ad43-a812e0d69573
# ╠═53b80d45-5eaa-4be8-a729-ee5e43477885
# ╠═104d1d6a-e6ed-4511-b08a-a72315959390
# ╠═0709ba9a-de5a-4e33-88f1-c10a49bfc065
# ╟─5b4cda6b-67e9-4c8a-8e7e-c6dd791f8726
# ╠═93d718df-351e-4111-99b5-b7ddaf657955
# ╠═96bb6208-c718-48e1-80d2-0e3f9fcc1127
# ╟─c10b27d7-c222-47fb-bbb2-3e55cc030e50
# ╟─27cf7e8b-3b5e-4401-a246-3a8949829764
# ╟─0800e2f6-4085-42dc-982c-e2b833b4171a
# ╟─8a2baef9-9ab3-4702-a3c2-af8300c83f7d
# ╟─6ee5026a-387b-4b30-acb7-303cb9da8724
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
# ╠═2ba1d475-e39b-4bee-8597-c468677a976d
# ╟─796be5ea-944b-4827-bfe6-654664c35fb3
# ╟─fba7d164-ff1c-4587-92e7-e7fd0668c0bd
# ╟─1f77d228-4106-4cc8-a9b3-05855f94660e
# ╟─5af768d8-c1f4-4721-bf14-bd43517e609c
# ╠═b14ce83b-ac6a-4baf-b499-16dedff13fa3
# ╠═2c7ac71d-2983-4da2-8ebf-748e51bd9d08
# ╠═fd3a09c5-6b02-4382-9eaf-fa81e4589057
# ╟─4c928cc0-71dc-4bd3-9486-0b1bfb7220d5
# ╠═404b7809-6718-424f-8aec-a8b2c35701eb
# ╠═17d6b9cf-1557-4b13-82cd-e642219ba8ac
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
# ╠═c6ea482d-431b-41c8-8478-3b795c59c18f
