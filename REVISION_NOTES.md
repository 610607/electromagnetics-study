# Revision notes: fourth → fifth edition / 第五版修订说明

The fourth edition was complete in content but read like an atlas: topics stood side by side, the same idea was derived in several places, and the reader was often told *what* holds before seeing *why* it must. The fifth edition keeps the physics and the numbers and rebuilds the order and the teaching around one principle given by the reader: **the goal is that people learn, not that topics are listed** (目的是让人学会而非罗列).

第四版内容完整，但更像一本图谱：专题并列，同一思想在多处重复推导，读者常常先被告知“什么成立”，后（或从未）看到“为什么必须如此”。第五版保留全部物理内容与数值，按“让人学会而非罗列”的原则重排顺序、重写讲解。

## 1. A progressive order / 循序渐进的顺序
- 24 chapters in 7 parts replace the old edition's 14 source files (22 lectures plus two appendices), which were organized by topic rather than by learning order. Each part answers one question before the next one needs it: the language of fields → electricity and magnetism side by side → time scales and potentials → the core field problems → finite elements → applications → synthesis and validation.
- Each mathematical identity is introduced once, where it is first needed, and numbered V1–V9 in that order (Appendix A lists where each is introduced and used).
- Duplicated material now has one home, and other chapters link to it: the two-element bar (Ch 16), the gauge example (Ch 9), the two-layer conductor (Ch 6), diffusion and skin depth (Ch 8), the Poynting theorem (Ch 7).
- A learning route with capability checkpoints opens the book; every chapter opens with its guiding question, learning goals and prerequisites (linked).

## 2. Similarities and differences made explicit / 突出概念之间的异同
- 81 "compare" boxes put two easily confused ideas side by side (what is the same, what differs, one sentence to remember), e.g. E vs D, B vs H, EQS vs MQS, one-way vs coupled MQS, essential vs natural boundary conditions, nodal vs edge elements, Picard vs Newton, secant vs differential permeability, soft vs hard magnetic materials, lumped circuit vs 3-D field model, concomitant field vs gradient nonlinearity, verification vs validation. Appendix C indexes all of them.
- Electricity and magnetism are taught as two parallel languages (Part II), with figures drawn in the same layout so the analogy and its limits are visible.

## 3. Teaching structure in every chapter / 每章的学习结构
Guiding question → intuition → derivation → worked examples with full solutions (73 in all) → common pitfalls → comparisons → takeaways → self-check with answers. Optional depth is marked ("deeper") so a first reading can skip it.

## 4. New material / 新增内容
- **Lumped magnetic circuits (Ch 14)**, requested for this edition: from field to network (assumptions stated), reluctance and magnetomotive force, Kirchhoff's laws for flux, nodal analysis, fringing and leakage, energy, inductance and force, nonlinear operating points with Newton's method, permanent magnets as Thevenin/Norton sources with load lines and (BH)max, a multi-pillar permanent-magnet yoke, and a validation study against 2-D nonlinear FEM that shows where the network is accurate and where it is not.
- Ferromagnetic materials (Ch 13): secant vs differential permeability and the incremental tensor, hysteresis and recoil, and why B–H interpolation must be shape-preserving (cubic spline vs PCHIP, computed).
- Magnets and yokes (Ch 15): demagnetizing factors from Osborn's closed forms, a computed nonlinear 2-D magnet cross-section, symmetry boundary conditions, Maxwell-stress forces.
- TES/TMS (Ch 10–11) on one finite-element mesh: electrode models compared (equipotential edge singularity vs complete electrode model), the induced field as a weighted projection, unit field maps.
- Eddy currents (Ch 12): a thin conducting-shell model for gradient-coil shields, the same result as a shorted secondary circuit, coil terminal voltage and energy.

## 5. Figures / 插图
- All figures redrawn from Python sources with one palette, one font family and semantic colours shared with the text; field lines are computed from flux functions, not sketched.
- Shaded 3-D figures (solenoid field, coil above tissue block, divergence and curl, interface probes, vector-potential circulation, multi-pillar magnet, edge element, gradient field, stream function on a cylinder) are computed in Python and rendered with three.js; labels are typeset afterwards with the book's fonts.
- Each figure exists in Chinese and English with identical geometry.

## 6. Traceability / 可追溯性
- Every number in the text is either from the fourth edition or recomputed and asserted by `checks/check_chNN.py`; teaching inputs are labelled as such.
- Methods are cited to real, linked sources (Haus & Melcher, Griffiths, Purcell & Morin, Fitzgerald et al., Osborn, Fritsch & Carlson, SimNIBS, MFEM, GetDP, FEMM, COMSOL, IT'IS, and the MRI and stimulation literature); no empirical rule is introduced without a source.
- `validate_source.py` checks that both editions share labels, citations, references, figures, teaching boxes and displayed mathematics.

## 7. Scope that remains outside the book / 仍不在本书范围内的内容
Advanced hysteresis models, coupled thermal physiology, mesh generation algorithms, detailed RF network design and neuron population models remain further courses. The worked models are teaching models; they do not replace validated anatomical data, calibrated coil waveforms, measured B–H data, production 3-D or full-wave solvers, or clinical and standards-based validation.

高级磁滞模型、热生理耦合、网格生成算法、射频网络设计与神经元群体模型仍属后续课程。书中的模型是教学模型，不能代替经过验证的解剖数据、校准波形、实测 B–H 数据、生产级三维或全波求解器，以及临床与标准层面的验证。
