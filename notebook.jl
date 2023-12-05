### A Pluto.jl notebook ###
# v0.19.32

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

# ╔═╡ e11f28b6-8f91-11ee-088d-d51c110208c6
begin
	using CairoMakie
	using DataStructures
	using Flux
	using GLPK
	using Graphs
	using GraphMakie
	using GraphMakie.NetworkLayout
	using JuMP
	using LinearAlgebra
	using PlutoTeachingTools
	using PlutoUI
	using ProgressLogging
	using Random
	using Test
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

# ╔═╡ 3f89b439-03e7-4e1e-89ab-63bbf5fa2194
n = 5; m = 5;

# ╔═╡ 0f4090e3-864c-46e5-bb28-203e735c63a8
g = Graphs.grid((n, m))

# ╔═╡ 7b17ae02-e713-4168-a288-d966c60043b7
c_range = 1:20

# ╔═╡ 5e867265-c733-485a-b39a-c4320e99c92a
begin
	Random.seed!(10)
	c = [rand(c_range) for _ in 1:ne(g)]
end

# ╔═╡ 5c261981-aa3c-474a-8e6e-b7a38ec02275
function compute_layout(g)
	[(i, j) for j in 1:m for i in 1:n]
end

# ╔═╡ e8574fa4-a6ad-442c-8fbe-42fe731a1e08
sqrt(25)

# ╔═╡ 81185112-2c81-4246-8128-2925345a9bc0
function plot_graph(graph, cost; grid=true, n=n, m=m)
	function compute_layout(g)
		[(i, j) for j in 1:m for i in 1:n]
	end

	f, ax, p = graphplot(
		graph;
		layout=grid ? compute_layout : Shell(),
		ilabels=fill(" ", nv(graph)),
		elabels=repr.(floor.(cost)),
		edge_width=3,
	)
	p.elabels_rotation[] = Dict(i => 0.0 for i in 1:ne(graph))
	p.elabels_offset[] = [Point2f(-0.02, 0.0) for i in 1:ne(g)]
	
	hidedecorations!(ax); hidespines!(ax)
	ax.aspect = DataAspect()
	autolimits!(ax)
	return f
end

# ╔═╡ 18e0aea4-5940-4bf3-b29b-9ec90c3108ba
plot_graph(g, c)

# ╔═╡ b4c0b7f5-8863-4921-915f-c7b73cb1e792
md"### Kruskal algorithm

The minimum weight spanning tree problem is polynomial, and can be solved efficiently using the kruskal algorithm:
"

# ╔═╡ b9daab11-d807-40fd-b94b-bc79ae80275e
function kruskal(g, weights; minimize=true)
    connected_vs = IntDisjointSets(nv(g))

	mst = falses(ne(g))

    edge_list = collect(edges(g))
	order = sortperm(weights; rev=!minimize)
	value = 0.0

	mst_size = 0

    for (e_ind, e) in zip(order, edge_list[order])
        if !in_same_set(connected_vs, src(e), dst(e))
            union!(connected_vs, src(e), dst(e))
			mst[e_ind] = true
			mst_size += 1
			value += weights[e_ind]
            (mst_size >= nv(g) - 1) && break
        end
    end

    return value, mst
end

# ╔═╡ 77630435-3536-4714-b4c7-db4473e7ba0e
md"Other option from the `Graphs.jl` package:"

# ╔═╡ 2251158b-d21a-4e4e-bc11-89bf7c385557
Graphs.kruskal_mst

# ╔═╡ 55ca5072-6831-4794-8aed-68d8b56f7f80
tree_value, T = kruskal(g, c)

# ╔═╡ 2f9c2388-a178-452b-a013-a2cc1cabc4b4
md"""### Visualization"""

# ╔═╡ 3258654a-f289-486c-b2c8-735828dad89f
function plot_forest(tree, graph, cost; grid=true, n=n, m=m)
	function compute_layout(g)
		[(i, j) for j in 1:m for i in 1:n]
	end

	f, ax, p = graphplot(
		graph;
		layout=grid ? compute_layout : Shell(),
		ilabels=fill(" ", nv(graph)),
		elabels=repr.(floor.(cost)),
		edge_color=[e ? :red : :black for e in tree],
		edge_width=[e ? 3 : 0.5 for e in tree],
	)
	p.elabels_rotation[] = Dict(i => 0.0 for i in 1:ne(graph))
	p.elabels_offset[] = [Point2f(-0.02, 0.0) for i in 1:ne(graph)]

	hidedecorations!(ax); hidespines!(ax)
	ax.aspect = DataAspect()
	autolimits!(ax)
	return f
end

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
	\text{s.t.}\quad & \sum_{e\in E} (y_e + z_{es}) = |V| - 1 & \forall s\in S\\
	& \sum_{e\in E(Y)} (y_e + z_{es}) \leq |Y| - 1,\quad &\forall \emptyset\subsetneq Y\subsetneq V,\forall s\in S\\
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
struct Instance
    graph::SimpleGraph{Int}
    first_stage_costs::Vector{Float64}
	second_stage_costs::Matrix{Float64}
end

# ╔═╡ e14e5513-5cc2-4b70-ab29-8ee53ca166cc
nb_scenarios(instance) = size(instance.second_stage_costs, 2)

# ╔═╡ c34c3f25-58ea-4219-b856-2ed9d790d291
function random_instance(; n, m, nb_scenarios=1, c_range=1:20, d_range=1:20, seed)
	g = grid((n, m))
	rng = MersenneTwister(seed)
	c = [rand(rng, c_range) for _ in 1:ne(g)]
	d = [rand(rng, d_range) for _ in 1:ne(g), _ in 1:nb_scenarios]

	return Instance(g, c, d)
end

# ╔═╡ b0155649-8f26-47ac-9d80-95a979f716cb
small_instance = random_instance(; n=3, m=3, nb_scenarios=1, seed=0)

# ╔═╡ c111dadd-3cb6-4cb0-b082-b67e11248e1c
S = 50

# ╔═╡ 8bc212ec-5a5d-401d-97d0-b2e0eb2b3b6f
instance = random_instance(; n, m, nb_scenarios=S, seed=0)

# ╔═╡ 76cbf0da-7437-464a-ba1b-e093cabd3b83
md"""### Visualization tools"""

# ╔═╡ bfcf2678-bc62-49a5-8856-a08c6a3a2f34
function plot_forest(tree, instance; grid=true, n=n, m=m)
	(; graph, first_stage_costs, second_stage_costs) = instance
	cost = Int.(min.(first_stage_costs, second_stage_costs[:, 1]))
	function compute_layout(g)
		[(i, j) for j in 1:m for i in 1:n]
	end

	f, ax, p = graphplot(
		graph;
		layout=grid ? compute_layout : Shell(),
		ilabels=fill(" ", nv(graph)),
		elabels=repr.(floor.(cost)),
		edge_color=[e ? :red : :black for e in tree],
		edge_width=[e ? 3 : 0.5 for e in tree],
	)
	p.elabels_rotation[] = Dict(i => 0.0 for i in 1:ne(graph))
	p.elabels_offset[] = [Point2f(-0.02, 0.0) for i in 1:ne(graph)]

	hidedecorations!(ax); hidespines!(ax)
	ax.aspect = DataAspect()
	autolimits!(ax)
	return f
end

# ╔═╡ 8339de9c-2dc5-41cf-8e93-cba341e0e6f6
plot_forest(T, g, c; grid=true)

# ╔═╡ 6186efdf-227e-4e95-b788-5dd3219162e7
begin
	scenario_slider = @bind current_scenario PlutoUI.Slider(1:S; default=1, show_value=true);
end;

# ╔═╡ a0cff617-ebe6-4fce-8f1f-a54200c18a39
function plot_forests(forest1, forest2, instance; grid=true, n=n, m=m)
	graph = instance.graph
	cost = Int.(instance.first_stage_costs)
	cost2 = Int.(@view instance.second_stage_costs[:, current_scenario])

	function compute_layout(g)
		[(i, j) for j in 1:m for i in 1:n]
	end

	tree = forest1 .|| @view forest2[:, current_scenario]
	colors = fill(:black, ne(graph))
	width = fill(0.5, ne(graph))
	colors[forest1] .= :red
	colors[@view forest2[:, current_scenario]] .= :green
	width[tree] .= 3
	labels = copy(cost)
	labels[.!forest1] .= cost2[.!forest1]
	
	f, ax, p = graphplot(
		graph;
		layout=grid ? compute_layout : Shell(),
		ilabels=fill(" ", nv(graph)),
		elabels=repr.(labels),
		edge_color=colors,
		edge_width=width,
	)
	p.elabels_rotation[] = Dict(i => 0.0 for i in 1:ne(graph))
	p.elabels_offset[] = [Point2f(-0.02, 0.0) for i in 1:ne(graph)]

	hidedecorations!(ax); hidespines!(ax)
	ax.aspect = DataAspect()
	autolimits!(ax)
	return f
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
- A BitVector representing the edges in set ``Y`` (`Y[e] == 1` if ``e \in E(Y)``)
- The number of vertices in ``Y``.
")

# ╔═╡ d9441eda-d807-4452-af10-11804bc668da
function MILP_separation_pb(graph, weights; MILP_solver=GLPK.Optimizer, tol=1e-5)
	return false, falses(ne(graph)), 0
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

# ╔═╡ 9f0431b8-7f5e-4081-bb09-c8c3014e035b
TODO(md"Implement this better formulation in the `cut_separation_pb` function.

This function must have three outputs in this order
- A boolean `found` telling if a constraint is violated (i.e. if a cut should be added)
- A BitVector representing the edges in set ``Y`` (`Y[e] == 1` if ``e \in E(Y)``)
- The number of vertices in ``Y``.
")

# ╔═╡ 7c1b96bc-b493-4b47-baef-22c6629b8286
function cut_separation_pb(graph, weights; MILP_solver=GLPK.Optimizer, tol=1e-5)
    return false, falses(ne(graph)), 0
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
	separate_constraint_function=MILP_separation_pb,
    MILP_solver=GLPK.Optimizer,
)
	# unpack fields
	(; graph, first_stage_costs, second_stage_costs) = instance
	S = nb_scenarios(instance)
	E = ne(graph)
	V = nv(graph)

	# Initialize model and link to solver
	model = Model(MILP_solver)

	# Add variables
    @variable(model, y[e in 1:E], Bin)
    @variable(model, z[e in 1:E, s in 1:S], Bin)

	# Add an objective function
	@expression(
		model,
		first_stage_objective,
		sum(first_stage_costs[e] * y[e] for e in 1:E)
	)
	@expression(
		model,
		second_stage_objective,
		sum(second_stage_costs[e, s] * z[e, s] for e in 1:E for s in 1:S) / S
	)
    @objective(model, Min, first_stage_objective + second_stage_objective)

	# Add constraints
	@constraint(
		model, [s in 1:S],
		sum(y[e] + z[e, s] for e in 1:E) == V - 1
	)

    call_back_counter = 0

    function my_callback_function(cb_data)
        # TODO
    end

	set_attribute(model, MOI.LazyConstraintCallback(), my_callback_function)

	optimize!(model)
	@info "Optimal solution found after $call_back_counter cuts"
    return objective_value(model), value.(y) .> 0.5, value.(z) .> 0.5
end

# ╔═╡ a129d5aa-1d45-407a-aeb2-00845330a0cb
md"""## 4. Testing"""

# ╔═╡ 0a6fc7ae-acb4-48ef-93ac-02f9ada0fcae
md"Now, we can apply the branch-and-cut on a small instance. However, it will struggle on larger ones because it's quite slow."

# ╔═╡ a01ba16a-1aa7-4bcf-8335-ba67815bfe87
vv, yy, zz = cut_generation(
	small_instance; separate_constraint_function=MILP_separation_pb, MILP_solver=GLPK.Optimizer
)

# ╔═╡ dff1db4f-b68b-410b-8c29-7bd30b8364b1
plot_forest(yy .|| zz[:, 1], small_instance; grid=true, n=3, m=3)

# ╔═╡ 0b861384-0c87-47f0-866f-b9dc616285a2
# vv_, yy_, zz_ = cut_generation(
	# instance; separate_constraint_function=MILP_separation_pb,  MILP_solver=GLPK.Optimizer
# )

# ╔═╡ 85fd6b13-421a-4834-b227-55bba9f12f24
# scenario_slider

# ╔═╡ 4bc6ca8f-0c88-48d3-8a36-5607951b0c86
# plot_forests(yy_, zz_, instance; grid=true)

# ╔═╡ 2c2b132c-6b66-4d7b-ad43-a812e0d69573
md"""The min-cut formulation being faster, we can apply to a larger instance:"""

# ╔═╡ 104d1d6a-e6ed-4511-b08a-a72315959390
cut_value, cut_y, cut_z = cut_generation(
	instance; separate_constraint_function=cut_separation_pb, MILP_solver=GLPK.Optimizer
)

# ╔═╡ 5b4cda6b-67e9-4c8a-8e7e-c6dd791f8726
scenario_slider

# ╔═╡ 93d718df-351e-4111-99b5-b7ddaf657955
plot_forests(cut_y, cut_z, instance; grid=true)

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
function column_generation(instance; MILP_solver=GLPK.Optimizer, tol=0.00001)
	S = nb_scenarios(instance)
	E = ne(instance.graph)
	# TODO
	return 0.0, zeros(S), zeros(E, S), [falses(E)]
end

# ╔═╡ 6d832dba-3f72-4782-919f-e1c1a0a92d3b
_, ν, μ, columns = column_generation(instance; MILP_solver=GLPK.Optimizer)

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

	return objective_value(model), value.(y) .> 0.5, value.(z) .> 0.5
end

# ╔═╡ 59bf6b13-0579-4f95-8acf-3c23d9bb9463
columns

# ╔═╡ 9e23f72a-0220-49d0-8f17-948cce8addbb
column_value, column_y, column_z =
	column_heuristic(instance, columns[1:min(length(columns), 50)])

# ╔═╡ 54aa7e04-897d-42d4-9ff9-62d8992397ec
scenario_slider

# ╔═╡ 3f94c697-bb57-4414-babe-74860ec0ac60
plot_forests(column_y, column_z, instance; grid=true)

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
\text{s.t.}\quad & -\nu_s - \sum_{e\in T}\mu_{es} \geq 0 & \forall T\in\mathcal{T}\\
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

# ╔═╡ 4bd8e5a0-f5a9-49a1-8ee2-32a1a225855d
TODO("Implement the benders decomposition described above")

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
\mathrm{s.t.} & 0\leq\mathbf{y} \leq M\\
& \mathbf{y}_s + \mathbf{z}_s \in \mathcal{P}, \quad\quad \text{for all $s$ in $S$}  
\end{array}
```
where ``M`` is a large constraint.
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
    inst::Instance,
    θ::AbstractMatrix,
    scenario::Int,
    grad::AbstractMatrix,
)
	(; graph, second_stage_costs) = inst

	weights = min.(-θ[:, scenario], second_stage_costs[:, scenario])

    value, tree = kruskal(inst.graph, weights)

	# update gradient
	slice = (-θ[:, scenario] .< second_stage_costs[:, scenario]) .&& tree
	grad[slice, scenario] .-= 1 / S

    return value ./ S
end;

# ╔═╡ ec163fa0-122c-4354-870a-c8d6370ef684
# Returns the value of the solution of `inst` with `forest` as first stage solution.
function evaluate_first_stage_solution(instance::Instance, forest)
	(; graph, first_stage_costs, second_stage_costs) = instance
    value = sum(first_stage_costs[forest])

	S = nb_scenarios(instance)
    values = zeros(S)
	forests = falses(ne(graph), S)
    Threads.@threads for s in 1:S
		weights = deepcopy(second_stage_costs[:, s])
        m = minimum(weights) - 1
        m = min(0, m - 1)
        weights[forest] .= m  # set weights over forest as the minimum

		# find min spanning tree including forest
        _, tree_s = kruskal(graph, weights)
		forest_s = tree_s .- forest
		forests[:, s] .= forest_s
		values[s] = dot(second_stage_costs[:, s], forest_s) / S
    end

    return value + sum(values), forests
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
    Threads.@threads for s in 1:S
		# Different part of grad are modified
        values[s] = second_stage_optimal_solution!(inst, θ, s, grad)
    end
    value += sum(values)
    return value, grad
end;

# ╔═╡ 2481e7f5-4db0-4db2-a4db-24512eb5d6df
md"""### 3. Lagrangian heuristic

Once a solution of the relaxed problem is found, we have one solution $y_s$ per scenario $s$. We can then use an heuristic to reconstruct a good first stage decision $y$.
This heuristic gives an upper bound at each iteration of the main algorithm, while the relaxation gives a lower bound.

Below is an example implementation of such an heuristic:
"""

# ╔═╡ f6dbbcb6-a16e-4630-ac74-06ce3d12e040
function lagrangian_heuristic(θ::AbstractMatrix; inst::Instance)
    # Retrieve - y_{es} / S from θ by computing the gradient
	(; graph) = inst
	S = nb_scenarios(inst)
    grad = zeros(ne(graph), S)
    Threads.@threads for s in 1:S
        second_stage_optimal_solution!(inst, θ, s, grad)
    end
    # Compute the average (over s) y_{es} and build a graph that is a candidate spannning tree (but not necessarily a spanning tree nor a forest)
    average_x = -vec(sum(grad; dims=2))
    weights = average_x .> 0.5
    # Build a spanning tree that contains as many edges of our candidate as possible
    _, tree_from_candidate = kruskal(graph, weights; minimize=false)
    # Keep only the edges that are in the initial candidate graph and in the spanning tree
    forest = weights .&& tree_from_candidate
	v, _ = evaluate_first_stage_solution(inst, forest)
    return v, forest
end;

# ╔═╡ 7934b593-36c3-4c70-b51b-aaf8f219aaf3
md"""### 4. Main algorithm"""

# ╔═╡ a6186708-a499-43bc-89e2-99a4abd0b700
TODO(md"Implement the full lagrangian relaxation using all the functions defined above.

The function has 5 outputs as a named tuple:
- `lb`: final lower bound
- `ub`: final upper bound
- `forest`: final forest solution
- `lb_history`: lower bound history
- `ub_history`: upper bound history
")

# ╔═╡ d07d3192-32bd-4f8d-8775-23429e888eb6
function lagrangian_relaxation(
    inst::Instance; nb_epochs=100, stop_gap=1e-4, show_progress=true
)
    θ = zeros(ne(inst.graph), nb_scenarios(inst))

    opt = Adam()

    lb = -Inf
    ub = Inf
    best_θ = θ
    forest = Edge{Int}[]

    last_ub_epoch = -1000

    lb_history = Float64[]
    ub_history = Float64[]
    @progress for epoch in 1:nb_epochs
        # TODO
		if (ub - lb) / abs(lb) <= stop_gap
			println("Stopped after ", epoch, " gap smaller than ", stop_gap)
			break
		end
    end

	ub, forest = lagrangian_heuristic(best_θ; inst=inst)
    return (; lb, ub, forest, lb_history, ub_history)
end;

# ╔═╡ e6f62957-7cba-48a1-bfd3-52e796c71fed
md"""### 5. Testing"""

# ╔═╡ 5529ac14-1171-4e77-b3b4-dbcde4b704a4
res = lagrangian_relaxation(instance; nb_epochs=25_000)

# ╔═╡ a75f1918-a7e3-4041-a4bf-b64fb7094546
begin
	f = Figure()
	ax = Axis(f[1, 1])
	lines!(ax, res.ub_history; label="Upper bound: lagrangian heuristic", color=:orange)
	lines!(ax, res.lb_history; label="Lower bound: lagrangian relaxation", color=:purple)
	axislegend(ax)
	f
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
GraphMakie = "1ecd5474-83a3-4783-bb4f-06765db800d2"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
CairoMakie = "~0.11.2"
DataStructures = "~0.18.15"
Flux = "~0.14.6"
GLPK = "~1.1.3"
GraphMakie = "~0.5.8"
Graphs = "~1.9.0"
JuMP = "~1.16.0"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
ProgressLogging = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "bab1bcceb9c5c1fc49b33acce67fb10a48c3c6b7"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "247efbccf92448be332d154d6ca56b9fcdd93c31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.6.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "0da671c730d79b8f9a88a391556ec695ea921040"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

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
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "fdbad23ab5b44ecd1aa77c84e110becfddf2d789"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.11.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "006cc7170be3e0fa02ccac6d4164a1eee1fc8c27"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.58.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
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

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

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
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

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
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

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

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "26eb8e2331b55735c3d305d949aabd7363f07ba7"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.11"

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

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a6c00f894f24460379cb7136633cef54ac9f6f4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.103"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArraysCore"]
git-tree-sha1 = "499b1ca78f6180c8f8bdf1cabde2d39120229e5c"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.6"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

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

[[deps.FastRounding]]
deps = ["ErrorfreeArithmetic", "LinearAlgebra"]
git-tree-sha1 = "6344aa18f654196be82e62816935225b3b9abe44"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.3.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "28e4e9c4b7b162398ec8004bdabe9a90c78c122d"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.8.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "b97c3fc4f3628b8835d83789b09382961a254da4"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.6"

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

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "50351f83f95282cf903e968d7c6e8d44a5f83d0b"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

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
version = "6.2.1+2"

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

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d53480c0793b13341c40199190f92c611aa2e93c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.2"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

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

[[deps.GraphMakie]]
deps = ["DataStructures", "GeometryBasics", "Graphs", "LinearAlgebra", "Makie", "NetworkLayout", "PolynomialRoots", "SimpleTraits", "StaticArrays"]
git-tree-sha1 = "8411166edd8b7fb02e9982710eedc944412b2c19"
uuid = "1ecd5474-83a3-4783-bb4f-06765db800d2"
version = "0.5.8"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

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

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "af13a277efd8a6e716d79ef635d5342ccb75be61"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

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

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "EnumX", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "f59e639916283c1d2e106d2b00910b50f4dab76c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.21.2"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "d65930fa2bc96b07d7691c652d701dcbe7d9cf0b"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SnoopPrecompile", "SparseArrays"]
git-tree-sha1 = "25b2fcda4d455b6f93ac753730d741340ba4a4fe"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.16.0"

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
git-tree-sha1 = "b0737cbbe1c8da6f1139d1c23e35e7cea129c0af"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.13"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "c879e47398a7ab671c782e02b51a4456794a7fa3"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

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

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Permutations", "Primes", "SimplePolynomials"]
git-tree-sha1 = "f45c914bba600a0a7b6acf39d9374ca44dd28220"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.3"

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

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "c165f205e030208760ebd75b5e1f7706761d9218"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.1"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

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
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Setfield", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "db14b5ba682d431317435c866734995a89302c3c"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.2"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "e81e6f1e8a0e96bf2bf267e4bf7f94608bf09b5c"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.1"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "362ae34a5291a79e16b8eb87b5738532c5e799ff"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.23.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

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

[[deps.Mods]]
git-tree-sha1 = "072c23c3d4b1f2f56501c0e9873bbe74231b508f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.0.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "ac86d2944bf7a670ac8bf0f7ec099b5898abcc09"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.8"

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

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "StaticArrays"]
git-tree-sha1 = "91bb2fedff8e43793650e7a677ccda6e6e6e166b"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.6"
weakdeps = ["Graphs"]

    [deps.NetworkLayout.extensions]
    NetworkLayoutGraphsExt = "Graphs"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

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
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

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
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4e5be6bb265d33669f98eb55d2a57addd1eeb72c"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.30"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "eed372b0fa15624273a9cdb188b1b88476e6a233"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.2"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4745216e94f71cb768d58330b059c9b76f32cb66"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.14+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "c7745750b8a829bc6039b7f1f0981bcda526a946"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.19"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

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
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PolynomialRoots]]
git-tree-sha1 = "5f807b5345093487f733e520a1b7395ee9324825"
uuid = "3a141323-8675-5d76-9d11-e1df1406c778"
version = "1.0.0"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "a9c7a523d5ed375be3983db190f6a5874ae9286d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.6"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "1d05623b5952aed1307bf8b43bec8b8d1ef94b6e"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.5"

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

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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
git-tree-sha1 = "6990168abf3fe9a6e34ebb0e05aaaddf6572189e"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.10"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

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

[[deps.SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "db0219befe4507878b1a90e07820fed3e62c289d"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.0"

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

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e9330391d04241eafdc358713b48396619c83bcb"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.1"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "91402087fd5d13b2d97e3ef29bbdf9d7859e678a"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.1"

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

[[deps.StableHashTraits]]
deps = ["Compat", "SHA", "Tables", "TupleTools"]
git-tree-sha1 = "d29023a76780bb8a3f2273b29153fd00828cb73f"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "1.1.1"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "5ef59aea6f18c25168842bded46b16662141ab87"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.7.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

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

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

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

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

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

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "155515ed4c4236db30049ac1495e2969cc06be9d"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.4.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "da69178aacc095066bad1f69d2f59a60a1dd8ad1"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.0+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

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

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5ded212acd815612df112bb895ef3910c5a03f57"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.67"

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

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

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
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

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
"""

# ╔═╡ Cell order:
# ╟─6cd3608b-0df6-4d22-af6c-9b6cb1d955c3
# ╠═e11f28b6-8f91-11ee-088d-d51c110208c6
# ╠═1b51af71-8474-46e6-91b2-35fc9adb2c5a
# ╠═30670737-22c2-42e1-a0a4-43aa0fa70752
# ╟─ab5c0cf3-6ce9-4908-8c09-b664503e5ac1
# ╟─bcc75aa2-9cd7-49bf-ad46-9dd5a8db3ef0
# ╟─4ca8f8f1-058f-4b47-adce-4cdbe916d628
# ╟─06b5c71a-bb44-4694-be43-1b3e9b38ece2
# ╠═3f89b439-03e7-4e1e-89ab-63bbf5fa2194
# ╠═0f4090e3-864c-46e5-bb28-203e735c63a8
# ╠═7b17ae02-e713-4168-a288-d966c60043b7
# ╠═5e867265-c733-485a-b39a-c4320e99c92a
# ╠═5c261981-aa3c-474a-8e6e-b7a38ec02275
# ╠═e8574fa4-a6ad-442c-8fbe-42fe731a1e08
# ╠═81185112-2c81-4246-8128-2925345a9bc0
# ╠═18e0aea4-5940-4bf3-b29b-9ec90c3108ba
# ╟─b4c0b7f5-8863-4921-915f-c7b73cb1e792
# ╠═b9daab11-d807-40fd-b94b-bc79ae80275e
# ╟─77630435-3536-4714-b4c7-db4473e7ba0e
# ╠═2251158b-d21a-4e4e-bc11-89bf7c385557
# ╠═55ca5072-6831-4794-8aed-68d8b56f7f80
# ╟─2f9c2388-a178-452b-a013-a2cc1cabc4b4
# ╟─3258654a-f289-486c-b2c8-735828dad89f
# ╟─eea94e99-cc33-4464-ac24-587466b17e48
# ╠═8339de9c-2dc5-41cf-8e93-cba341e0e6f6
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
# ╟─76cbf0da-7437-464a-ba1b-e093cabd3b83
# ╟─bfcf2678-bc62-49a5-8856-a08c6a3a2f34
# ╟─a0cff617-ebe6-4fce-8f1f-a54200c18a39
# ╟─6186efdf-227e-4e95-b788-5dd3219162e7
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
# ╠═7c1b96bc-b493-4b47-baef-22c6629b8286
# ╟─60f15817-f696-4fa4-a69d-46c00f2513c7
# ╟─ec711cc2-1613-42d7-bdae-01460509da24
# ╟─4b952721-31ba-4583-9034-3a5aaec07934
# ╠═154b4c51-c471-490e-8265-230f3eda92e4
# ╟─a129d5aa-1d45-407a-aeb2-00845330a0cb
# ╟─0a6fc7ae-acb4-48ef-93ac-02f9ada0fcae
# ╠═a01ba16a-1aa7-4bcf-8335-ba67815bfe87
# ╠═dff1db4f-b68b-410b-8c29-7bd30b8364b1
# ╠═0b861384-0c87-47f0-866f-b9dc616285a2
# ╠═85fd6b13-421a-4834-b227-55bba9f12f24
# ╠═4bc6ca8f-0c88-48d3-8a36-5607951b0c86
# ╟─2c2b132c-6b66-4d7b-ad43-a812e0d69573
# ╠═104d1d6a-e6ed-4511-b08a-a72315959390
# ╟─5b4cda6b-67e9-4c8a-8e7e-c6dd791f8726
# ╠═93d718df-351e-4111-99b5-b7ddaf657955
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
# ╟─796be5ea-944b-4827-bfe6-654664c35fb3
# ╟─fba7d164-ff1c-4587-92e7-e7fd0668c0bd
# ╟─1f77d228-4106-4cc8-a9b3-05855f94660e
# ╟─4bd8e5a0-f5a9-49a1-8ee2-32a1a225855d
# ╟─74fc70cc-42e7-4e20-9c04-cebe2dcbd3f3
# ╟─78ec76dd-db39-4ad0-8db5-559839420d96
# ╟─aee74231-afe1-4793-b12a-89948473b6fb
# ╠═2ca79687-187a-4259-9470-65d59e537749
# ╟─da57e0b6-e917-4d86-87d4-07aeac0bbdb2
# ╠═afaf3780-52bc-452b-b966-6b4f59166e66
# ╠═ec163fa0-122c-4354-870a-c8d6370ef684
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
