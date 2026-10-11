# Fifth edition — chapter plan, content allocation and label registry

Source of the old (4th) edition (not included in this repository): the Chinese lectures `zh/*.tex`, the English
lectures `chapters/*.tex` and the figure scripts `make_*.py` of the fourth-edition source package.
Numerics carried over unchanged: `numerics.py`, `concept_checks.py`, `extended_checks.py`.

Chinese chapters live in `zh/chNN_slug.tex`. Finished exemplar chapter: `zh/ch05_matter.tex` (read it before writing anything).

The author's request: make the course progressive (循序渐进) and easier to learn, with special
attention to the **similarities and differences between concepts**, and make the whole book and
its figures more beautiful and intuitive. The author added: **"目的是让人学会而非罗列"** — the goal is
that a reader *learns*, not that topics are listed. The author also asked for a **lumped magnetic
circuit** chapter (Ch 14 below).

Old-edition shorthand used below: `atlas:k` = `zh/02_atlas.tex` subsection k (k-th `\subsection`),
`conn:k` = `03_connections.tex`, `form:k` = `02_formulations.tex`, `mat:k` = `04_materials_regimes.tex`,
`mag:k` = `03_magnetics.tex`, `fem:k` = `04_fem.tex`, `app:k` = `05_applications.tex`,
`rf:k` = `09_rf_sar.tex`, `proj:k` = `10_learning_projects.tex`, `prac:k` = `06_practice.tex`,
`wc:k` = `08_worked_checks.tex`, `calc:k` = `01_calculus.tex`, `pic:k` = `00_picture.tex`.
(k counts `\subsection`s from 1 in each file, across `\lecture`s.)

Every piece of old content must land somewhere (or be a deliberate merge of a duplicate).
Duplicates in the old edition are merged into ONE home (listed below); other chapters refer to it.

---------------------------------------------------------------------------------------------------
## Part I 场的语言 (Foundations)

### Ch 1 `ch:intro` 一个问题引出整门课  — file `zh/ch01_intro.tex`
Core question: TMS 线圈外（组织里）的磁场与静磁场相比微不足道，组织里为什么仍会有足以刺激神经的电场？要算出它，需要知道什么？
- 1.1 一个具体场景：线圈、导体、看不见的场。Qualitative causal chain: coil current → B → changing flux → induced E → charges move → surface charges → secondary field; magnetic materials respond; energy stored/dissipated (pic:1). Same picture for an MRI gradient coil near a body.
- 1.2 两个基本场：用力来定义 E 与 B (Lorentz force, eq:lorentz) (pic:2 first part).
- 1.3 Maxwell 方程：先认识，后理解. Show the four macroscopic equations (eq:gaussboth, eq:curlboth) **read in words**, say D, H are introduced in Ch 5 (for now read them as ε0E and B/μ0); table/figure mapping each equation to the chapter that builds it. Charge conservation mentioned as consequence (detail in Ch3/5).
- 1.4 一个模型由四层组成：定律、材料、表示、观测量 (pic:1 second half, pic:3; whole_problem fig). Potentials introduced only as "another description" with promise of Ch 9 (pic:2 second half, very short).
- 1.5 本书的路线与贯穿全书的算例 (running examples: parallel-plate capacitor, two-layer conductor, long solenoid, gapped core/magnetic circuit, conducting square with an inclusion (TMS/TES), two-element bar (FEM)).
- keyidea "遇到新符号时应当回答的问题" (pic end) → turn into an intuition or takeaway.
Figures: `intro_scene` (new), `whole_problem` (redesign), `maxwell_map` (new: four equations, each with a pictogram + chapter where it is built).

### Ch 2 `ch:calculus` 场的几何：梯度、散度与旋度 — `zh/ch02_calculus.tex`
Core question: 一张矢量场图里，“源头”与“漩涡”分别在哪里？怎样用数学语言说出来？
- 2.1 `sec:calc-grad` 梯度：等高线与“下坡”；gradient theorem φ(b)−φ(a)=∫∇φ·dℓ (label eq:gradthm) (calc:1 part).
- 2.2 `sec:calc-div` 散度：小盒子的净流出；derive divergence theorem intuitively (V1) id:div.
- 2.3 `sec:calc-curl` 旋度：小回路的环流；Stokes (V2) id:stokes.
- 2.4 对照：散度与旋度 (two examples F=(x,y,0), (−y,x,0); plus uniform field (both zero) and saddle (x,−y) (both zero but not uniform) → "local equations alone do not determine a field; boundary information does" (conn:3 first paragraphs). compare box.
- 2.5 两个恒为零的组合 (V3) id:zeros → why potentials exist (preview E=−∇φ, B=∇×A) ; Helmholtz idea in one paragraph (conn:3).
- 2.6 乘积法则 (V4) id:product: material coefficients cannot be pulled out of derivatives.
- (Topology "curl-free ≠ conservative" moves to Ch 9; interface conditions move to Ch 6; continuity to Ch 3; V5–V9 are introduced where first used — see identity table.)
Figures: `gradient_contours` (new), `div_curl_four` (new 4-panel: source, vortex, uniform, saddle, each with a small flux box and circulation loop overlay and its div/curl values).

---------------------------------------------------------------------------------------------------
## Part II 电与磁：两套平行的语言

### Ch 3 `ch:electric` 电荷、电场与电势 — `zh/ch03_electric.tex`
Core question: 电场的“强”和电势的“高”是一回事吗？电荷静止与流动时，场各遵守什么规律？
- 3.1 电荷与电荷密度 (atlas:1): q, Q, ρ, q_s; geometry; free vs bound only previewed (Ch 5).
- 3.2 电场：力、方向与做功 (atlas:2): work, field lines ≠ trajectories.
- 3.3 高斯定律：电荷是电场的源 (vacuum form ∇·(ε0E)=ρ, integral form, parallel-plate field; label eq:gaussvac).
- 3.4 `sec:electric-potential` 静电场无旋，所以有电势 (atlas:3 first half): eq:atlasphi; reference/constant; worked check wc:1 (large potential ≠ large field) as example.
- 3.5 对照：电势与电场 (compare box: φ vs E: scalar/vector, reference-dependent/not, units, what measures it, which enters force).
- 3.6 `sec:electric-current` 电流与电荷守恒 (atlas:4): J, I, carriers, conventional current; continuity derived by counting charge in a box → eq:continuity (∂tρ+∇·J=0, integral form); steady ∇·J=0 ≠ J=0; sheet current K and the different units (A/m² vs A/m vs A); wc:4 (net-neutral conductor still carries current/heats) as example. (calc:3 merged here; the derivation "from Ampère–Maxwell" lives in Ch 5.)
- 3.7 `sec:electric-ohm` 导体中的欧姆定律 (atlas:4 end + atlas:7 first half + mat:2): J_c=σE (eq:jfrome), source vs response split J_f=J_s+J_c; resistor R=ℓ/(σA) briefly (full system in Ch7); power density E·J; anisotropic σ tensor (eq:sigmatensor), example σ=diag(1,4), E=(1,1); diffusion-MRI caveat.
- 3.8 对照：电荷密度与电流密度 (charge_current figure).
Figures: `charge_current` (redesign), `potential_vs_field` (new 2-panel; replaces electric_potential), `anisotropic_current` (redesign).

### Ch 4 `ch:magnetic` 电流、磁场与感应 — `zh/ch04_magnetic.tex`
Core question: 为什么理想螺线管外面没有磁场，绕着它的导线圈里却能感应出电压？
- 4.1 磁力与磁通密度 B (atlas:8): q v×B does no work; flux Φ_B (eq:atlasflux), orientation.
- 4.2 磁场没有源头：∇·B=0 (atlas:8 end) — contrast with Gauss's law (compare box: ∇·E vs ∇·B).
- 4.3 电流是磁场的“涡旋源”：安培定律 (vacuum ∇×B=μ0J, integral form, long wire, long solenoid B inside; label eq:amperevac).
- 4.4 Biot–Savart 与圆环线圈 (form:3 result only, derivation in Ch 9; form:4 loop benchmark, eq:loopaxis; label eq:biotsavart; return path must be complete).
- 4.5 `sec:magnetic-A` 磁矢势 A：环量就是磁通 (atlas:12 first part: eq:atlasA, eq:Aflux, uniform B example with A_sym; "A arrows are not B arrows"). Gauge freedom only previewed → Ch 9.
- 4.6 `sec:magnetic-faraday` 法拉第定律：变化的磁通产生环流电场 (eq:faradayintegral, EMF eq:atlasemf, uniform dB/dt example form:5 eq:uniforminduction with numbers 1000 T/s, r=0.08 m → 40 V/m; "two meanings of gradient" remark form:5 end).
- 4.7 `sec:solenoid` 螺线管：外部 B=0 而 E≠0 (conn:8 full with numbers; eq:solenoidA, eq:solenoidE; wc:6; topology remark).
- 4.8 对照：静电场与感应电场 (both are E, same force; sources charges vs changing flux; curl zero vs nonzero; potential exists vs not; field lines open vs closed).
- 4.9 `sec:magnetic-linkage` 磁通、磁链与电感 (atlas:15; λ vs NΦ; sign conventions EMF vs terminal voltage; L defined, λ=LI; preview V=RI+dλ/dt).
- 4.10 对照：电流 I 缩放 B，电流变化率 dI/dt 缩放感应 E (short; detail Ch 11).
Figures: `magnetic_flux` (redesign), `loop_axis` (redesign, inset geometry), `a_uniform` (new from a_symmetric: circulating A, uniform B, loop circulation = flux), `induction` (redesign), `solenoid_profiles` (redesign: cross-section + radial profiles), `static_vs_induced` (new comparison 2-panel), `flux_linkage` (redesign).

### Ch 5 `ch:matter` 物质中的电与磁 — DONE (`zh/ch05_matter.tex`)
Covers atlas:5,6,9,10,11(first half), conn:2, mat:1, pic:2 (P/M, D/H), wc:3, wc:5, displacement current, full Maxwell table.
Labels defined there: see registry.

### Ch 6 `ch:interfaces` 界面：什么连续，什么跳变 — `zh/ch06_interfaces.tex`
Core question: 电流从高电导率组织流进低电导率组织时，电场为什么会突然变大？界面上到底堆积了什么？
- 6.1 两种“探针”：薄盒与窄回路 (calc:4 derivation).
- 6.2 四条界面条件 eq:normalbc, eq:tangentbc + charge continuity at interface eq:jnormal (calc:4).
- 6.3 对照总表：E, D, J, B, H 哪个分量连续 (table + compare box normal vs tangential).
- 6.4 `sec:interfaces-conductivity` 电导率界面：电场“折射” (eq:einterface, anisotropic case).
- 6.5 双层导体：电流连续，电场跳变 — the ONE home of the two-layer example (merge conn:7 two-layer part, form:10, wc:4 second paragraph): numbers 8 mm/2 mm, 0.2/0.02 S/m, 1 mA → J=10 A/m², E=50/500 V/m, ΔV=1.4 V, R=1400 Ω, P=1.4 mW; fixed I vs fixed E heating "paradox". Label ex:twolayer.
- 6.6 `sec:interfaces-surfacecharge` 界面电荷：次级电场的来源 (conn:7 last paragraph: continuous J_n does not mean zero surface charge; preview TMS).
- 6.7 磁界面：B_n 与 H_t (calc:4 magnetic part; field lines entering iron nearly normal; nonlinear materials use the same Maxwell interface laws).
- 6.8 局部方程 + 边界条件 = 唯一的场 (short: why BCs are part of the physics; artificial truncation boundaries are physical choices).
Figures: `interface_probes` (new), `interface` (redesign: left conductivity interface with E and J; right permeability interface with B and H), `conduction_profiles` (redesign, includes potential; replaces old `slab`).

### Ch 7 `ch:systems` 四个标准系统：电容、电阻、电感与磁路 — `zh/ch07_systems.tex`
Core question: 电路课里的 R、C、L 是从哪里来的？它们把场的哪些信息“压缩”掉了？
- 7.1 电容器：一次连起 Q、D、E、V、C (conn:6 with numbers; fixed Q vs fixed V energy; eq:capall). The displacement-current argument is in Ch5 — refer to it.
- 7.2 电阻：V、E、J、R 与热 (conn:7 first part; power three ways).
- 7.3 电感：螺线管的 L 与磁能 (atlas:16 inductor part; L≈μN²A/ℓ; multi-coil mutual L preview).
- 7.4 `sec:systems-magcircuit` 磁路初识：铁芯与气隙 (conn:10: eq:coreconnection NI = ℓ_y H_y + g B_g/μ0; why most MMF drops in the gap; L = N²/ℛ preview). Full theory → Ch 14.
- 7.5 `sec:systems-energy` 能量：储存、耗散与流动 (atlas:17, conn:11): w_e, w_m, p_loss; **Poynting theorem derived here** with (V5) id:crossdiv → eq:poynting (merge app:4 derivation here; Ch 18 refers back); resistor example S inward = E·J (poynting_resistor); peak vs RMS 1/2.
- 7.6 对照：R、C、L 与磁阻 ℛ (table: which field relation, units, energy stored vs dissipated, what is lost in the compression).
- 7.7 阻抗与频域预告 (atlas:16 end: Z, R+jωL, 1/(jωC); differential inductance for nonlinear cores).
Figures: `four_systems` (new 2×2), `poynting_resistor` (redesign), `core_gap` (redesign of core_connection, simple gapped ring core).

---------------------------------------------------------------------------------------------------
## Part III 时间、近似与表示

### Ch 8 `ch:regimes` 快与慢：静态、准静态与波 — `zh/ch08_regimes.tex`
Core question: 同样是“低频”，为什么身体里可以忽略感应电流的磁场，铜屏蔽里却不行？什么时候必须回到完整的波动方程？
- 8.1 准静态不等于静态 (mat:5 first paragraph).
- 8.2 `sec:regimes-phasor` 相量：把 ∂t 变成 jω (atlas:19, mat:3): eq:phasor, peak vs RMS, eq:admittivity (Y=σ+jωε, double counting losses), IT'IS data caveat.
- 8.3 三把尺子：三个时间尺度 (mat:4 charge relaxation eq:relaxation; magnetic diffusion time; transit time) → eq:scales (η_w, η_d, η_m, η_w²=η_dη_m), which length ℓ, rise time vs repetition rate.
- 8.4 `sec:regimes-eqs` EQS：保留电荷，忽略感应 (mat:6, eq:eqs_general).
- 8.5 `sec:regimes-mqs` MQS：保留感应，忽略位移反馈 (mat:7, eq:mqs; ωε/σ meaningless in air).
- 8.6 `sec:regimes-oneway` 单向耦合：MQS 中的再一步近似 (mat:7 end).
- 8.7 对照：EQS 与 MQS (+ static, full-wave) — table of kept terms; figure regime_terms.
- 8.8 磁扩散与趋肤深度 — ONE home (form:18 derivation with (V6) id:curlcurl → eq:diffusion, skin depth eq:skindepth; mat:8 numbers table tissue vs copper at 3 kHz: δ=12.995 m vs 1.207 mm, η_m=4.74e-4 vs 5.50).
- 8.9 波动方程从哪里来 (mat:9, eq:wave, k=β−jα, good-conductor limit).
- 8.10 对照：低频、近场、准静态 (mat:10, eq:nearfar, 128 MHz example, keyidea three labels).
Figures: `regime_terms` (new: which terms each regime keeps), `timescales` (optional), `skin_depth` (redesign), `diffusion` (redesign), `near_far` (redesign).

### Ch 9 `ch:potentials` 势函数：同一个场的不同“坐标” — `zh/ch09_potentials.tex`
Core question: 两个看起来完全不同的矢势图，怎么可能描述同一个实验？
- 9.1 为什么用势：把约束写进未知量 (form:1, conn:5): eq:vectorpotential, eq:Etwopotentials.
- 9.2 `sec:potentials-gauge` 规范自由度：表示不唯一，物理唯一 — ONE home of the uniform-B two-gauge example (merge atlas:12 second half, form:2, conn:5 end, wc:2): eq:gauge, time-dependent case with φ compensation, unit test for TMS source; Coulomb gauge; comparing solvers/surrogates needs gauge-invariant quantities.
- 9.3 库仑规范与 Biot–Savart 的推导 (form:3 derivation: eq:curlcurlA, eq:Agreen, refer to eq:biotsavart in Ch4).
- 9.4 `sec:potentials-scalar` 磁标势与拓扑 (atlas:13, form:6, calc:5): H=−∇ψ, eq:magscalar, Laplace for shims, wire/cut, "curl-free ≠ conservative".
- 9.5 `sec:potentials-2d` 二维与轴对称：A_z 等值线就是磁力线 (atlas:12 subsubsection, rA_θ flux function).
- 9.6 `sec:potentials-stream` 表面流函数：生成电流而不是场 (atlas:14; eq:stream definition K_s=∇_sψ_s×n; current continuity; contours = current paths; Ch 20 uses it).
- 9.7 对照：四种“势” φ、A、ψ、ψ_s (big compare + figure four_potentials).
Figures: `gauge_pair` (redesign of a_symmetric+a_alternative), `potential_bridge` (redesign), `az_contours` (redesign), `scalar_cut` (redesign), `stream_function` (redesign), `four_potentials` (new).

---------------------------------------------------------------------------------------------------
## Part IV 核心场问题

### Ch 10 `ch:tes` TES：给定电流的体积传导 — `zh/ch10_tes.tex`
Core question: 两块电极贴在头皮上，电流在头里怎样分布？哪些边界条件真正决定了答案？
- 10.1 推导电导方程 (form:7): eq:tespde, complex eq:tescomplex, dispersion.
- 10.2 边界：符号、电流相容性与参考 (form:8): eq:tesneumann, eq:currentcompatibility, reference vs grounding (compare box: fixing a reference vs imposing φ=0 on the boundary).
- 10.3 电极不只是一个电流标签 (form:9): eq:cem, three electrode models (compare), edges/singularities.
- 10.4 叠加与导联场 (form:11): eq:tesleadfield; linear in source, not in anatomy; |E| not linear.
- (Two-layer example lives in Ch 6 — refer to ex:twolayer.)
Figures: `tes_setup` (new, computed with numerics P1 FEM: square with inclusion, two electrode patches, equipotentials + current lines), `electrode_models` (new schematic of three electrode models).

### Ch 11 `ch:tms` TMS：给定磁源的感应 — `zh/ch11_tms.tex`
Core question: 头部组织几乎不改变线圈的磁场，为什么却能显著改变电场？
- 11.1 从线圈矢势到次级势方程 (form:12, conn:9): eq:tmsE, eq:tmspde, eq:tmsboundary (not homogeneous Neumann).
- 11.2 为什么 B 几乎不变，E 却明显改变 (form:13, conn:9 end): surface charges; induction_correction figure.
- 11.3 修正是一个加权投影 (form:14): **introduce (V7) id:scalaribp here**; eq:tmsfunctional, eq:tmsorthogonal; tms_fem_inclusion.
- 11.4 离散算子与源线性 (form:15): eq:tmsoperator.
- 11.5 波形、dI/dt 与单位场图 (form:4 end eq:epcoil, form:16 eq:tmswaveform, pulse; wc:7 drive_scaling with numbers).
- 11.6 对照：TES 与 TMS (same K, different RHS and BC; scaling with I vs dI/dt; figure tes_tms_compare).
Figures: `induction_correction` (redesign, computed), `tms_fem_inclusion` (redesign, computed), `tes_tms_compare` (new, computed on same mesh), `pulse` (redesign), `drive_scaling` (redesign).

### Ch 12 `ch:eddy` 涡流与耦合 MQS — `zh/ch12_eddy.tex`
Core question: 梯度线圈旁边的金属冷屏为什么会“拖慢”梯度？什么时候感应电流必须反过来改写磁场？
- 12.1 何时感应电流必须反馈 (form:17): eq:av_mqs, eq:av_charge; stranded vs solid conductors (compare).
- 12.2 用扩散的眼光看涡流 (refer to Ch 8 diffusion; time constants of a plate/shield; delay of gradient) — keep to content traceable to Ch 8 results.
- 12.3 时域与谐波系统 (form:19).
- 12.4 线圈端电压、储能与互感 (form:20): eq:circuit, eq:multicircuit with numbers (20 µH, 0.1 Ω, 5000 A, 50 A/µs → 500 V, 1000 V, 250 J).
- 12.5 运动导体 (form:21).
- 12.6 对照：单向感应与耦合涡流.
Figures: `eddy_oneway_vs_coupled` (new schematic), optional `coil_voltage` (bar of resistive/inductive voltage).

### Ch 13 `ch:bh` 铁磁材料：非线性与磁滞 — `zh/ch13_bh.tex`
Core question: “这块铁的磁导率是多少”为什么常常是一个没有唯一答案的问题？
- 13.1 B–H 曲线是本构关系，不是场图 (mag:1): eq:syntheticbh (B_s=1.6 T, H_s=800 A/m).
- 13.2 饱和不是硬上限 (mag:1, atlas:11 end).
- 13.3 割线、微分与增量磁导率 (mag:2, atlas:11): eq:permeabilities, sech² derivative; preview of the incremental tensor (parallel ↔ 1/μ_diff, transverse ↔ 1/μ_sec; derivation in Ch 17); compare box (three permeabilities).
- 13.4 `sec:bh-hysteresis` 磁滞、剩磁与矫顽力 (mag:3): B_r, H_cB vs H_cJ (compare), loss = loop area; **recoil line of a permanent-magnet material** eq:recoil (from mag:6 first part) so Ch 14 can use it.
- 13.5 把标量曲线用于矢量场 (mag:4): eq:HofB.
- 13.6 导入测量数据时避免创造虚假材料 (mag:5): units gauss/oersted, monotone interpolation, extrapolation.
- 13.7 对照：非线性、磁滞、剩磁与小信号 (mag keyidea).
Figures: `permeabilities` (new: secant, tangent, and incremental parallel/transverse inset; replaces bh_secant_tangent), `hysteresis` (redesign), `bh_interpolation` (new, computed: sparse synthetic B–H samples, unconstrained cubic spline vs PCHIP, showing overshoot / negative slope).

### Ch 14 `ch:lumped` 集总磁路：从场到网络 — `zh/ch14_lumped.tex`  (NEW, requested by the author)
Core question: 一个铁芯加气隙的结构，怎样不解三维场就估出气隙磁场、电感和吸力？这种估计在什么时候可信？
Content (derive everything from Maxwell; cite Fitzgerald/Umans Ch.1 and Haus & Melcher Ch. 9 — keys `fitzgerald`, `haus`):
- 14.1 从场到集总：可以压缩的条件 (flux confined in high-μ paths, gap ≪ cross-section, quasistatic, leakage small) — when it fails.
- 14.2 元件：磁动势 𝓕=NI、磁阻 ℛ=ℓ/(μA)、磁导 𝒫=1/ℛ、磁通 Φ (eq:mmf, eq:reluctance).
- 14.3 两条“基尔霍夫定律”：节点磁通守恒 from ∇·B=0 (eq:kcl-flux)；回路磁动势平衡 from Ampère (eq:kvl-mmf).
- 14.4 串联、并联与节点法 (worked example: E-core / three-limb core network solved by nodal analysis; numbers computed and checked by script).
- 14.5 气隙：边缘效应与漏磁 (fringing raises effective area — calibrate the factor A_eff/A with 2-D/3-D FEM or measurement, do NOT invent empirical rules; leakage = parallel permeance paths; why flux has no insulator: μ_iron/μ_0 ~ 10³–10⁵ vs σ_metal/σ_air ≳ 10²⁰).
- 14.6 电感、能量与吸力 from the network: L=N²/ℛ_total (eq:Lreluctance), mutual inductance, W = Φ²ℛ/2 = 𝓕²/(2ℛ) (linear), force F = −½Φ² dℛ/dx at fixed flux (= +½𝓕² d𝒫/dx at fixed MMF) — tie to virtual work in Ch 18.
- 14.7 `sec:lumped-nonlinear` 非线性磁阻与工作点 (conn:10 + mag:8): graphical load line on the yoke B–H curve and Newton iteration; uses eq:syntheticbh.
- 14.8 `sec:lumped-pm` 永磁体进入磁路 (mag:8–10): two equivalent sources from the recoil line (eq:recoil): MMF source 𝓕_c=H_c ℓ_m with internal reluctance ℓ_m/(μ0μ_rec A_m) (Thevenin) **vs** flux source Φ_r=B_r A_m with parallel permeance (Norton) — compare box; single-loop magnet/gap/yoke derivation eq:nonlinearmagcircuit; **the old numerical example** (Φ=1.60702e-3 Wb, B_g=0.803511 T, B_y=1.607023 T, H_y=5590.68 A/m, H_m=−3.00491e5 A/m; ideal yoke 0.845070 T, −4.92%) — from numerics.magnetic_circuit(); load line eq:loadline and (BH)_max.
- 14.9 多柱磁轭算例 (teaching example, computed): two magnet blocks / pole pieces with N identical return pillars in parallel; show B_g vs number of pillars and pillar cross-section, including yoke saturation with the synthetic curve; state clearly which effects (leakage, fringing, pole-face non-uniformity) the network omits.
- 14.10 `sec:lumped-validity` 集总模型何时可信：与 FEM 对照标定 (verify each assumption; compare gap flux and leakage with a 2-D/3-D field solution; **a lumped/analytic model must be validated before it is embedded in an optimizer or surrogate** — pitfall box).
- 14.11 对照：电路与磁路 (table: what maps — V↔𝓕, I↔Φ, R↔ℛ, G↔𝒫, KCL/KVL; what doesn't — no flux insulator → leakage; reluctance stores energy rather than dissipates; nonlinearity/saturation; fringing).
Figures: `circuit_analogy` (new), `gapped_core_network` (new: geometry ↔ network), `three_limb_network` (new), `fringing_leakage` (new schematic with flux paths), `nonlinear_operating_point` (new, computed), `pm_equivalents` (new: Thevenin vs Norton), `load_line` (redesign), `multipillar_yoke` (new, computed: geometry sketch + B_g vs pillars curves).
A check script `checks/check_ch14.py` must reproduce every number used in the chapter.

### Ch 15 `ch:magnets` 永磁体、磁轭与 MRI 磁体：走出集总模型 — `zh/ch15_magnets.tex`
Core question: 磁路给出了气隙场的估计，为什么真正的 MRI 磁体仍要做三维场计算？还需要算出哪些东西？
- 15.1 永磁体不是高磁导率区域 (mag:6): eq:pmconstitutive tensor form, material axes, software conventions (remanence/magnetization/coercivity/equivalent surface current); compare soft vs hard magnetic material.
- 15.2 退磁场：几何进入材料响应 (mag:7): eq:demag, M = χ/(1+N_dχ) H_applied; link to the sphere in Ch 5 (N_d=1/3); optionally a computed demag-factor curve for spheroids (Osborn 1945, key `osborn` — real: J. A. Osborn, Phys. Rev. 67, 351 (1945)).
- 15.3 磁体–磁轭 FEM 模型要解析什么 (mag:11): outputs list, air domain, symmetry.
- 15.4 受力：在应力定律明确的区域积分 (mag:12, atlas:18): eq:maxwellstress, torque, magnetic pressure B²/(2μ0).
- 15.5 对照：集总磁路与三维场模型 (what each gives; when to use which).
Figures: `magnet_assembly` (redesign of magnetic_circuit: schematic yoke magnet with pole pieces, leakage), `demag_factor` (new, computed from Osborn's closed forms for spheroids).

---------------------------------------------------------------------------------------------------
## Part V 从方程到矩阵：有限元

### Ch 16 `ch:femscalar` 标量有限元：从守恒到矩阵 — `zh/ch16_femscalar.tex`
Core question: 一条偏微分方程怎样变成一个可以求解、可以检验的矩阵方程？
- 16.1 `sec:femscalar-bar` 先算一个最小的例子：两单元导电棒 — ONE home (proj:2; eq:barassembled, eq:barpower), fully by hand; boundary reactions = terminal currents.
- 16.2 弱形式：用测试函数表达守恒 (fem:1; restate (V7) id:scalaribp from Ch 11; eq:scalarweak; compare: essential vs natural BC).
- 16.3 线性基函数与单元矩阵 (fem:2; eq:elementmatrix, eq:tetraK).
- 16.4 组装、边界与参考 (fem:3).
- 16.5 完整电极模型的弱形式 (fem:4; eq:cemweak).
- 16.6 节点解守恒什么、不守恒什么 (fem:5).
- 16.7 收敛实验 (fem:6; table + convergence figure; compare: algebraic residual vs discretisation error).
Figures: `two_element_bar` (redesign), `hat_functions_1d` (new), `nodal_basis` (redesign), `convergence` (redesign; data from numerics.run_all).

### Ch 17 `ch:femvector` 矢量有限元与非线性求解 — `zh/ch17_femvector.tex`
Core question: 为什么三维磁场不能当作三个独立的标量问题来解？非线性的铁怎样进入牛顿迭代？
- 17.1 为什么需要棱单元 (fem:7; eq:edgebasis; exact sequence) — compare nodal vs edge elements.
- 17.2 磁弱形式与边界符号 (fem:8; **introduce (V8) id:curlibp here**; eq:magweak; PM right-hand side).
- 17.3 规范自由度就是离散零空间 (fem:9).
- 17.4 二维与轴对称可用节点单元 (fem:10).
- 17.5 推导一致的牛顿切线 (fem:11; eq:newtonK, eq:inctensor boxed).
- 17.6 Picard、延拓与收敛证据 (fem:12; compare Picard vs Newton; optional computed comparison on the nonlinear magnetic-circuit equation).
- 17.7 偏置附近的线性源响应 (fem:13; eq:biaslinearization).
Figures: `edge_vs_nodal` (new, computed basis functions on a triangle), `exact_sequence` (new diagram), `picard_newton` (optional, computed). Ready-made 3-D figure (editor): `edge_element_3d` (nodal N0 colour on a tetrahedron vs edge function W01 arrows) — include it.

---------------------------------------------------------------------------------------------------
## Part VI 应用：同一场，不同问题

### Ch 18 `ch:framework` 统一框架与共有观测量 — `zh/ch18_framework.tex`
Core question: TES、TMS、MRI 梯度和磁体设计，到底哪些是共同的，哪些是各自特有的？
- 18.1 差异在源与观测量，不在定律 (app:1, app:2, conn:13): three operator families, eq:wholeoperator, superposition, application_map.
- 18.2 前向、逆向与响应 (app:3).
- 18.3 非线性磁能与共能 (app:5) (Poynting itself is in Ch 7 — refer to eq:poynting).
- 18.4 电感、互易性与电路能量 (app:6; virtual work fixed I vs fixed λ compare).
- 18.5 `sec:framework-sar` 从局部损耗到 SAR (app:7; eq:localsar; peak vs RMS numbers 2.5 vs 5 W/kg).
- 18.6 线性场，二次加热 (app:8; Q-matrix eq:sarQ, cross terms; figure sar_cross_terms with extended_checks numbers).
- 18.7 `sec:framework-reciprocity` 互易性与伴随灵敏度 (app:9).
- 18.8 同一场解，多种读出 (conn:13 end; field_observables).
Figures: `application_map` (redesign), `field_observables` (redesign), `sar_cross_terms` (new, computed).

### Ch 19 `ch:mri` MRI 的磁场观测量 — `zh/ch19_mri.tex`
Core question: 能不能随心所欲地设计一个只有 B_z = G x 的梯度场？
- 19.1 MRI 里不止一个电磁问题 (app:10).
- 19.2 Maxwell 方程约束所有可实现的梯度 (app:11: eq:roiMaxwell, eq:gradientmatrix, eq:xgradient, eq:zgradient; scalar-potential generation; spatial-gradient tensor G_ij vs ∂tB from atlas:18).
- 19.3 伴随场与梯度非线性 (app:12: eq:concomitant; compare).
- 19.4 从场图到相位与 k 空间 (app:13).
- 19.5 调和匀场与均匀性指标 (app:14; compare RMS vs peak-to-peak).
- 19.6 有限场图本身不能确定什么 (app:15).
Figures: `gradient_maxwell` (redesign: x- and z-gradient panels), `concomitant` (new, computed from eq:concomitant with stated teaching numbers). Ready-made 3-D figure (editor): `gradient_3d` (y=0 plane coloured by b_z = G_x x, arrows of the full b = (G_x z, 0, G_x x)) — include it.

### Ch 20 `ch:coils` 线圈逆向设计 — `zh/ch20_coils.tex`
Core question: 已经知道想要什么场，怎样找出一个真的能绕出来、又不会烧掉的线圈？
- 20.1 先设计源空间 (app:16; compare forward vs inverse).
- 20.2 流函数保证电流连续 (app:17; refers to sec:potentials-stream; contour extraction).
- 20.3 一致的场、电阻与电感算子 (app:18).
- 20.4 正则化与精确约束解 (app:19; eq:coilobjective).
- 20.5 主动屏蔽、匀场与硬件约束 (app:20).
- 20.6 FEM、边界方法与学习算子 (app:21).
Figures: `inverse_design_flow` (new diagram), optional computed `tradeoff_curve`. Ready-made 3-D figure (editor): `stream_cylinder_3d` (illustrative transverse-gradient stream function ψ_s = cos φ · z·exp(−z²/2s²) on a cylinder; contours = current paths with direction, four "fingerprints") — include it, labelled as illustrative, not an optimized design.

### Ch 21 `ch:neuro` 从场到神经响应 — `zh/ch21_neuro.tex`
Core question: 电场最强的地方，为什么不一定是神经最先兴奋的地方？
- 21.1 场解还不是神经响应 (app:22).
- 21.2 量纲一致地推导电缆方程 (app:23; eq:cablephysical, eq:cablenormalized).
- 21.3 激活函数及其符号 (app:24; eq:activating; V_e^eff along path).
- 21.4 弯曲与末端 (app:25; **introduce (V9) id:path here**; eq:curvatureAF).
- 21.5 主动膜与阈值搜索 (app:26).
- 21.6 强度–时程：被动精确解与经验拟合 (app:27; eq:exactsd, eq:lapicque; compare).
Figures: `activating_function` (redesign), `curved_axon` (new, computed), `cable_compartments` (new schematic), `strength_duration` (redesign).

### Ch 22 `ch:rf` 准静态的边界：射频场与 SAR — `zh/ch22_rf.tex`
Core question: RF 线圈就贴在身体上，为什么“近场”不足以让我们用准静态模型？
- 22.1 同一 MRI 系统的多个频率范围 (rf:1).
- 22.2 恢复频域场方程 (rf:2; eq:rfwave, eq:rfweak; uses id:curlibp).
- 22.3 发射观测量是旋转分量 (rf:3; eq:corotating; factor-2 trap).
- 22.4 `sec:rf-normalization` 比较 SAR 前先归一化源 (rf:4).
- 22.5 `sec:rf-averages` 三种平均 (rf:5; eq:rfQ, eq:rftimeaverage; compare RF-cycle vs spatial vs pulse-train averaging).
Figures: `rotating_components` (new, computed), `rf_averaging_ladder` (redesign).

---------------------------------------------------------------------------------------------------
## Part VII 综合与验证

### Ch 23 `ch:projects` 一体化仿真项目 — `zh/ch23_projects.tex`
- 23.1 `sec:projects-contract` 模型约定 (proj:1).
- 23.2 项目一：同一网格上的 TES 与 TMS (proj:3).
- 23.3 项目二：同一源基，多种读出 (proj:4).
- 23.4 项目三：先求非线性偏置，再求小信号 (proj:5; eq:biascheck).
- 23.5 项目四：同一电场，PNS 与加热 (proj:6; eq:nervedifference, eq:discretecable).
- 23.6 项目五：重新检查范围后再加入 RF (proj:7).
- 23.7 从场数组到可复现接口 (wc:8).
(Stage 1 two-element bar lives in Ch 16 — refer to sec:femscalar-bar.)
Figures: `pns_sar_branches` (redesign), `project_ladder` (new).

### Ch 24 `ch:practice` 验证、不确定性与可辩护的决策 — `zh/ch24_practice.tex`
- 24.1 从物理问题选择流程 (prac:1).
- 24.2 数值验证与物理确认 (prac:2; compare verification vs validation).
- 24.3 各应用的检查清单 (prac:3).
- 24.4 常见失误 (prac:4).
- 24.5 `sec:practice-errorbudget` 误差预算 (prac:6).
- 24.6 `sec:practice-reproducible` 可复现结果 (prac:7).
- 24.7 下一步学什么 (prac:5 + REVISION_NOTES scope boundaries).
Figures: `verification_validation` (new diagram).

### Appendices — `zh/appendices.tex`
- A `app:identities` 向量恒等式速查 (V1–V9 with where each is introduced and used).
- B `app:symbols` 符号与单位 (old 07 appendix B, updated).
- C `app:compare` 对照索引 (auto list `\listofcomparisons`) — written by the editor.

---------------------------------------------------------------------------------------------------
## Vector identities (number by first use; each defined ONCE with \tag{V#}\label{id:...})
| tag | label | statement | defined in |
|---|---|---|---|
| V1 | id:div | divergence theorem | Ch 2 |
| V2 | id:stokes | Stokes theorem | Ch 2 |
| V3 | id:zeros | curl grad = 0, div curl = 0 | Ch 2 |
| V4 | id:product | product rules for div and curl | Ch 2 |
| V5 | id:crossdiv | ∇·(F×G) = G·∇×F − F·∇×G | Ch 7 |
| V6 | id:curlcurl | curl curl = grad div − Laplacian | Ch 8 |
| V7 | id:scalaribp | scalar integration by parts | Ch 11 |
| V8 | id:curlibp | curl integration by parts | Ch 17 |
| V9 | id:path | chain rule along a path | Ch 21 |
Write them exactly as in old `01_calculus.tex` (same mathematics), e.g.
`\begin{equation} ... \tag{V7}\label{id:scalaribp}\end{equation}`.

## Label registry (labels other chapters may reference)
Only these labels (plus labels inside your own chapters) may be referenced across chapters.
Figure labels are always `fig:<figure file name>` (e.g. `fig:dipoles_pair`), and any `fig:<name>` of a figure that a chapter includes may be referenced from anywhere (e.g. `fig:coil_block_3d`, `fig:intro_scene`, `fig:multipillar_3d`). Ch 5 is the exception
only where noted.

```
ch:intro ch:calculus ch:electric ch:magnetic ch:matter ch:interfaces ch:systems ch:regimes
ch:potentials ch:tes ch:tms ch:eddy ch:bh ch:lumped ch:magnets ch:femscalar ch:femvector
ch:framework ch:mri ch:coils ch:neuro ch:rf ch:projects ch:practice
app:identities app:symbols app:compare
eq:lorentz eq:gaussboth eq:curlboth
sec:calc-grad sec:calc-div sec:calc-curl eq:gradthm id:div id:stokes id:zeros id:product
sec:electric-potential sec:electric-current sec:electric-ohm eq:gaussvac eq:atlasphi eq:continuity eq:jfrome eq:sigmatensor
sec:magnetic-A sec:magnetic-faraday sec:solenoid sec:magnetic-linkage eq:atlasflux eq:amperevac eq:biotsavart eq:loopaxis eq:atlasA eq:Aflux eq:atlasemf eq:faradayintegral eq:uniforminduction eq:solenoidA eq:solenoidE
sec:matter-dipoles sec:matter-P sec:matter-M sec:matter-DH sec:matter-analogy sec:matter-linear sec:matter-displacement sec:matter-maxwell eq:boundcharge eq:boundcurrent eq:materialdefinitions eq:freeDH eq:DHintegral eq:atlassphere eq:susceptibility tab:maxwell ex:dielectriccap ex:ironcore
eq:normalbc eq:tangentbc eq:jnormal eq:einterface sec:interfaces-conductivity sec:interfaces-surfacecharge ex:twolayer
sec:systems-magcircuit sec:systems-energy eq:capall eq:coreconnection eq:poynting id:crossdiv
sec:regimes-phasor sec:regimes-eqs sec:regimes-mqs sec:regimes-oneway eq:phasor eq:admittivity eq:relaxation eq:scales eq:eqs_general eq:mqs eq:skindepth eq:diffusion eq:wave eq:nearfar id:curlcurl
sec:potentials-gauge sec:potentials-scalar sec:potentials-2d sec:potentials-stream eq:vectorpotential eq:Etwopotentials eq:gauge eq:curlcurlA eq:Agreen eq:magscalar eq:stream
eq:tespde eq:tescomplex eq:tesneumann eq:currentcompatibility eq:cem eq:tesleadfield
eq:tmsE eq:tmspde eq:tmsboundary eq:tmsfunctional eq:tmsorthogonal eq:tmsoperator eq:tmswaveform eq:epcoil id:scalaribp
eq:av_mqs eq:av_charge eq:circuit eq:multicircuit
eq:syntheticbh eq:permeabilities eq:HofB eq:recoil sec:bh-hysteresis
eq:mmf eq:reluctance eq:kcl-flux eq:kvl-mmf eq:Lreluctance eq:nonlinearmagcircuit eq:loadline sec:lumped-nonlinear sec:lumped-pm sec:lumped-validity
eq:pmconstitutive eq:demag eq:maxwellstress
sec:femscalar-bar eq:scalarweak eq:elementmatrix eq:tetraK eq:cemweak eq:barassembled eq:barpower
eq:edgebasis eq:magweak eq:newtonK eq:inctensor eq:biaslinearization id:curlibp
eq:wholeoperator eq:localsar eq:sarQ sec:framework-sar sec:framework-reciprocity
eq:roiMaxwell eq:gradientmatrix eq:xgradient eq:zgradient eq:concomitant
eq:coilobjective
eq:cablephysical eq:cablenormalized eq:activating eq:curvatureAF eq:exactsd eq:lapicque id:path
eq:rfwave eq:rfweak eq:corotating eq:rfQ eq:rftimeaverage sec:rf-normalization sec:rf-averages
sec:projects-contract eq:biascheck eq:nervedifference eq:discretecable
sec:practice-errorbudget sec:practice-reproducible
eq:solenoidB eq:wireB eq:Afromflux eq:linkage eq:inductance eq:divB
eq:pdipole eq:mdipole eq:dipoletorque eq:Pdef eq:Mdef eq:allsourcegauss eq:allsourceampere eq:continuityfromampere sec:matter-puzzle sec:matter-puzzle-solved
eq:Enjump eq:refraction eq:hinterface tab:interfaces-summary
eq:Lsolenoid eq:magohm eq:energydensity eq:lossdensity eq:poyntingintegral eq:impedance tab:systems-rclr ex:systems-core ex:systems-inductor
front:colours front:notation sec:projects-p1 sec:projects-p2 sec:projects-p3 sec:projects-p4 sec:projects-p5 sec:projects-interface eq:projects-frames eq:projects-operators tab:projects-contract tab:projects-readouts tab:projects-metadata ex:projects-frame ex:projects-p1 ex:projects-p2 ex:projects-p3 ex:projects-p4 ex:projects-p5 ex:projects-units sec:practice-workflow sec:practice-vv sec:practice-checklists sec:practice-failures sec:practice-next eq:practice-indicator tab:practice-check-fields tab:practice-check-responses tab:practice-failures tab:practice-sources tab:practice-record tab:practice-next ex:practice-workflow ex:practice-budget ex:practice-twomesh sec:app-conventions sec:app-nine sec:app-ibp sec:app-family sec:app-fields sec:app-materials sec:app-lumped sec:app-scales sec:app-applications sec:app-collisions sec:app-constants tab:app-identities tab:app-collisions tab:app-constants
eq:framework-leadfield eq:framework-adjoint eq:framework-virtualwork eq:framework-Lij eq:framework-wm eq:framework-superposition eq:framework-sarmass sec:framework-energy sec:framework-inductance sec:framework-quadratic sec:framework-readouts ex:framework-coenergy ex:framework-force ex:framework-twochannel ex:framework-adjoint tab:framework-tasks sec:mri-maxwell sec:mri-concomitant sec:mri-kspace sec:mri-shim sec:mri-limits eq:mri-k eq:mri-rms eq:mri-bias ex:mri-concomitant ex:mri-homogeneity tab:mri-problems sec:coils-stream sec:coils-operators sec:coils-regularization sec:coils-hardware sec:coils-methods eq:coils-operators eq:coils-exact eq:coils-shim ex:coils-biplanar ex:coils-shield tab:coils-methods
sec:neuro-gap sec:neuro-cable sec:neuro-activating sec:neuro-bends sec:neuro-active sec:neuro-sd sec:rf-ranges sec:rf-equations sec:rf-rotating eq:neuro-veff eq:neuro-endbc eq:neuro-nodal eq:neuro-curvature eq:neuro-chronaxie eq:neuro-weiss eq:rf-flip eq:rf-basis ex:neuro-gauss ex:neuro-bend ex:neuro-sd ex:rf-flip ex:rf-quad ex:rf-units tab:rf-ranges
sec:femscalar-weak sec:femscalar-basis sec:femscalar-assembly sec:femscalar-cem sec:femscalar-conservation sec:femscalar-convergence tab:femscalar-convergence ex:femscalar-tet ex:femscalar-twolayer ex:femscalar-cembar ex:femscalar-thinlayer eq:femscalar-strong eq:femscalar-system sec:femvector-edge sec:femvector-weak sec:femvector-gauge sec:femvector-2d sec:femvector-newton sec:femvector-picard sec:femvector-bias ex:femvector-picard eq:femvector-pm eq:femvector-az
eq:bh-inc eq:bh-sech eq:bh-work tab:bh-permeabilities tab:bh-elements eq:magnets-chieff eq:magnets-shapeload eq:magnets-rotate tab:magnets-conventions tab:magnets-outputs ex:magnets-force ex:lumped-pillars ex:lumped-pm
sec:tes-boundary sec:tes-electrodes sec:tes-leadfield ex:tes-ground sec:tms-projection sec:tms-operator sec:tms-vs-tes ex:tms-square sec:eddy-diffusion ex:eddy-shell ex:eddy-voltage sec:eddy-circuit sec:eddy-motion
sec:regimes-scales sec:regimes-diffusion sec:regimes-wave sec:regimes-labels tab:regimes-3khz ex:regimes-scales ex:regimes-phasorsum ex:regimes-rf sec:potentials-biot sec:potentials-compare ex:potentials-gauge tab:potentials-four
ex:electric-cap ex:electric-layer ex:electric-phiE ex:electric-aniso eq:electric-resistance eq:electric-power eq:electric-plates eq:electric-coulomb eq:electric-contint tab:intro-examples
```
