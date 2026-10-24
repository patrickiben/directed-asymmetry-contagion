# Following the source of a cascade

A plain-language companion to *Anticipating Network Contagions: When Directedness Outperforms Connectedness*, Patrick Iben, draft dated 5 July 2026.

## The question you should carry through the paper

A place can be badly affected without being the main source of spread. The paper asks when a model gains more by supporting a transmitter than by supporting its worst-hit unit.

Its intervention results come from simulations fitted to real data. They do not measure the effects of real emergency, public-health or financial interventions.

This guide is for readers comfortable with averages and graphs, but new to network models. Read the lessons in order. Try each question before reading its answer. Page numbers below refer to the linked draft, not this guide.

[Read the source draft](../manuscript/anticipating_network_contagions_draft.pdf).

## D1. A source and a casualty can be different places

**Where we are**

Introduction, PDF pages 1–2; Figure 1, page 14.

**What the passage is doing**

It separates the allocation question from the observation that a network is under stress. A node is one modeled unit, such as a state or market. Interference means that what happens at one unit can affect others.

**In ordinary language**

Imagine a quiet upstream unit passing stress to a noisy downstream unit. Helping the noisy unit may leave the upstream stream of stress intact. That is a reason to investigate the source, not an instruction to deny help to the casualty.

**Evidence status**

Definition and research question. The quiet-source story is an explanation of the proposed mechanism, not evidence that every network has that structure.

**Do not accidentally conclude**

“Worst-hit” refers to the modeled stress series. It is not an independent measure of suffering, clinical need or the social value of assistance.

**Try it**

A downstream town reports the greatest damage. Does that identify the best source-control target? **Answer:** No. Damage and modeled transmission are different quantities. Identifying a source still requires data and a suitable model.

## D2. Turn time series into a transparent stand-in

**Where we are**

“Pipeline,” page 3; Methods, pages 9–10.

**What the passage is doing**

A non-negative first-order vector autoregression, or VAR(1), predicts each unit's next stress from the previous values of all units. Its coefficient matrix is Φ. Non-negative means its fitted coefficients cannot be negative. Ridge regularization penalizes large coefficients to stabilize estimation.

The fitted system is a surrogate: a simplified stand-in. Generalized forecast error variance decomposition, or GFEVD, divides prediction-error variance into contributions associated with different units' shocks. “To” summarizes exports, “from” summarizes imports, and net connectedness is to minus from. The largest positive net value identifies the transmitter.

**In ordinary language**

The model learns which earlier readings help predict later readings elsewhere. It then asks which unit sends out more predictive influence than it receives. This is a map of fitted prediction relationships, not a map of verified physical causes.

**Evidence status**

Defined statistical construction fitted to empirical series. A teaching example, not a paper result: exports of 6 and imports of 2 give a net value of 4, in the same stated units.

**Do not accidentally conclude**

A “Granger-causal” or forecast-variance direction is not a measured intervention effect. Omitted drivers, sampling and model choice can affect the direction. The paper does not identify a potential-outcomes treatment effect.

**Try it**

Could a unit with a modest current reading still be the transmitter? **Answer:** Yes. Its role depends on fitted exports minus imports, not on having the largest current reading.

## D3. Total connectedness does not tell you the direction

**Where we are**

“Connectedness does not predict controllability,” pages 3–4; Figure 2, page 14; Figure 4, page 15.

**What the passage is doing**

Total connectedness summarizes the overall amount of forecast-variance sharing. Directedness concerns its asymmetry. The paper compares smoke at about 82% total connectedness with flight delays at about 77%. These are nearly matched, not identical, connectedness values.

**In ordinary language**

Two networks can be strongly linked for different reasons. One can have a clear upstream source. Another can move together without a useful source-to-receiver gap. A single overall score hides that difference.

**Evidence status**

Empirical fitted-network comparison, followed by model-based control experiments. In the smoke surrogate, transmitter targeting reduces the cascade by about 10%, versus about 4% for the reactive rule. The flight-delay controllers remain near baseline.

**Do not accidentally conclude**

The smoke point estimate is not a measured reduction in pollution or a separately decisive statistical result. Its block-bootstrap advantage interval includes zero.

**Try it**

A new network has high total connectedness. Is that enough to predict a targeting advantage? **Answer:** No. Inspect the direction, the transmitter–casualty gap, uncertainty and the control test.

## D4. Remove direction and ask whether the advantage survives

**Where we are**

“Isolating the asymmetry,” page 3; “Interdiction and the symmetrization null,” page 10, equation 3.

**What the passage is doing**

The symmetrization null blends Φ with its symmetric projection, (Φ + Φᵀ)/2. The blend removes the antisymmetric part as its direction parameter falls to zero. Before each control simulation, the matrix is rescaled to the common spectral radius.

The spectral radius, ρ, is the largest absolute eigenvalue of the matrix. Here it controls the surrogate's tendency to amplify stress. The control experiments use a mildly amplifying regime, ρ = 1.05 or 1.06.

**In ordinary language**

The test gradually makes each modeled direction resemble its reverse. It then repeats the same allocation contest. If the source-targeting edge disappears, that supports a role for direction within this model.

**Evidence status**

An analytic matrix operation and an in-surrogate experiment. The symmetric matrix part is held fixed by the blend before rescaling. This is not a claim that every connectedness statistic stays numerically fixed along the path.

**Do not accidentally conclude**

The test does not remove direction from a real epidemic or financial system. It tests a mechanism in the fitted simulation. It does not prove a universal control theorem.

**Try it**

Why rescale each blended matrix rather than only the original? **Answer:** Otherwise a changing growth tendency could explain the performance change, instead of direction alone at the chosen growth setting.

## D5. Read a large result with its model qualifier attached

**Where we are**

“The clearest case,” page 4; Limitations, pages 8–9.

**What the passage is doing**

The equity example gives the clearest contrast in the draft. In the calibrated 2008 equity surrogate, transmitter targeting reduces the cascade by approximately 42%; reactive targeting gives approximately −7%. The negative reduction means a larger modeled cascade than no action.

**In ordinary language**

In this simulation, the two allocation rules can have very different effects. A rule aimed at the worst current reading can even make the modeled outcome worse. The result belongs to this fitted and rescaled experiment.

**Evidence status**

Model-based effect sizes from empirical calibration. Equity and the held-out COVID-19 network are the only cases whose individual block-bootstrap advantage intervals exclude zero.

**Do not accidentally conclude**

These percentages are not effects of bailouts, actual emergency triage or a proposed trading strategy. The intervention, budget, model and chosen growth regime are part of the result.

**Try it**

Would “source targeting cut the real equity crash by 42%” be faithful? **Answer:** No. Replace “real equity crash” with “cascade in the calibrated equity surrogate.”

## D6. Compare strong alternatives within the same experiment

**Where we are**

“Benchmark,” page 4; Figure 3, page 15.

**What the passage is doing**

The paper compares reactive control with correlation centrality, several directed scores and model predictive control. Correlation centrality ranks nodes by undirected association. Model predictive control, or MPC, repeatedly plans support using forecasts of the model.

On the held-out COVID-19 benchmark, directed methods give 35–65% surrogate cascade reductions. Correlation centrality gives about 5%, reactive triage 0.2%, and cross-entropy MPC about 30%. Other source-structure methods are also competitive on the anchor networks.

**In ordinary language**

The point is not that one named score wins every contest. Several methods that use source structure perform well in this benchmark. A more elaborate planner is not automatically better than a good simple diagnosis.

**Evidence status**

A model-based benchmark with competing methods. The caption's 0.2% reactive result is small, even though the plotted rounded label appears as zero.

**Do not accidentally conclude**

Do not combine these percentages with those from Figures 5 or 6. Those figures use different protocols. Their within-experiment comparisons, rather than pooled magnitudes, carry the argument.

**Try it**

If another directed method beats the named connectedness score, does that alone refute the paper's central pattern? **Answer:** No. The broader comparison concerns source-aware versus reactive targeting, not a monopoly for one estimator.

## D7. Keep failed predictions in the record

**Where we are**

“Six out-of-domain tests,” pages 5–6; Figure 4, page 15; prediction-first and availability statements, pages 10–11.

**What the passage is doing**

It distinguishes a held-out COVID-19 test from five prediction-first domain tests. Including COVID-19 gives three confirmations, one refinement and two failed predictions. The 2008 equity case is separate from these six tests.

| Prediction-first case | Reported verdict | What changed the interpretation |
|---|---|---|
| Asian currency crisis | Confirmation | Quiet hypothesized origin differs from the loudest receiver |
| Wildfire smoke | Confirmation | Transmitter and loudest receiver differ |
| Influenza | Refinement | Transmitter and loudest unit coincide |
| Flight delays | Failed prediction | Expected source structure is near-symmetric instead |
| Armed conflict | Failed prediction | Expected origin receives; the fitted system is weakly coupled |

**In ordinary language**

A useful test can reveal that your forecast about a domain was wrong. Keep that failed forecast visible. Then ask separately whether the fitted structure fits the narrower pattern you proposed.

**Evidence status**

A written prediction-first record and empirical model evaluations. The original five predictions lacked an externally verified public timestamp before analysis. A later public deposit proves the records existed by publication, not when they were first written.

**Do not accidentally conclude**

The five original tests are not externally preregistered studies. Three different prospective predictions were later registered at OSF; this draft does not analyze them. Relabeling the two failures as successful prior predictions would erase part of the result.

**Try it**

Which is the better summary: “all domain predictions succeeded” or “the failures helped define the observed scope”? **Answer:** The second. The domain predictions failed twice even if their fitted structures fall in the proposed null region.

## D8. Understand what the uncertainty checks can establish

**Where we are**

Discussion, pages 7–9; statistics, page 10; Figure 4 caption, page 15.

**What the passage is doing**

The simulations share fixed seeds across controllers. The main effect summaries use 16 seeds. A stationary block bootstrap resamples time-series blocks, rather than treating every observation as independent; the draft uses 300 resamples for 95% advantage intervals.

The transmitter matches the hypothesized historical origin in five of seven networks. The draft reports a probability near 10⁻⁴ under uniform random node selection, and about 1.5 × 10⁻³ under its restricted comparison. These are tests of specified recovery nulls.

**In ordinary language**

A positive average is not enough to make a result precise. Repeated samples can give a wide range of advantages. A small chance-recovery probability also does not prove that the fitted arrows are real causes.

**Evidence status**

Model-based uncertainty and null comparisons. The draft's five-test sign test gives p = 0.031; its confirm/refine-versus-falsify permutation comparison gives p = 0.10. These answer different questions in a small set of cases.

**Do not accidentally conclude**

Five recoveries are not seven independent intervention trials. Historical-origin matching depends on the proposed origins and the random-selection null. Most individual advantage intervals still include zero.

**Try it**

Can a positive average and a confidence interval containing zero both be correct? **Answer:** Yes. The average describes the sample result; the interval indicates substantial uncertainty about its advantage.

## D9. A fixed transmitter can be enough

**Where we are**

“When anticipation is needed,” pages 6–7; Figure 5, page 16.

**What the passage is doing**

A static rule chooses a transmitter once. A dynamic planner repeatedly chooses support. On a near-stationary synthetic surrogate, the static rule captures about 96% of the planner's reduction. In the separate COVID-19 decomposition, static targeting gives about 32%, versus 28% for MPC.

**In ordinary language**

If the important source stays put, repeatedly planning may add little. The hard part may be finding that source correctly. Extra machinery has to show a benefit in a fair comparison.

**Evidence status**

A separate surrogate decomposition experiment. “Stationary” here concerns a stable useful targeting structure; it does not mean every measured series is constant.

**Do not accidentally conclude**

The 32% result does not replace the 35–65% range in Figure 3. Different experimental protocols produced them. Nor does a result on these surrogates show that static policies always suffice.

**Try it**

What should a complex controller be compared with? **Answer:** A strong, simple source-aware rule under the same model, budget and protocol, not only a weak reactive baseline.

## D10. A moving transmitter changes the comparison

**Where we are**

Non-stationary systems, page 7; Figure 6, page 16.

**What the passage is doing**

Rolling windows refit the network using successive stretches of past data. In Figure 6, eight states take the leading role; the window-specific choice differs from the full-sample choice in 72% of windows. That is not a claim that the identity changes at every adjacent window in that fraction.

The adaptive surrogate rule uses past windows and gives about 8% reduction, versus 3% for the fixed full-sample rule and 10% for the oracle-window comparison.

**In ordinary language**

A source chosen from a long historical average may no longer be the current source. Updating the diagnosis can help in the moving-source experiment. The update must use information available at the time.

**Evidence status**

Observed changes in fitted roles and model-based controller results. An oracle comparison is an idealized benchmark, not automatically an achievable real-time policy.

**Do not accidentally conclude**

Neither a shifting fitted role nor an adaptive simulation proves a useful real-world intervention. Avoid using future observations to define a supposedly past-only action.

**Try it**

Why is a past-only update important? **Answer:** A controller with future information would have an unfair advantage and would not represent a realizable forecast procedure.

## D11. Forecasting real observations is a different test

**Where we are**

Raw-data out-of-sample evaluation, page 7.

**What the passage is doing**

The paper also predicts aggregate downstream stress on held-out real observations, rather than simulating support. It reports an 8% forecast-error reduction for the equity example. Across the three stationary-transmitter systems, two have a positive lead and the mean reduction is 3.3%; the sign-test p-value is 0.50.

**In ordinary language**

A model can make a useful-looking forecast comparison without proving that an intervention works. Here the raw-data forecast evidence is also too uncertain to count as a clear confirmation. The paper calls it suggestive support.

**Evidence status**

An empirical held-out forecasting result, reported as statistically non-significant. It is separate from surrogate intervention outcomes.

**Do not accidentally conclude**

The 3.3% mean is not a confirmed general predictive gain, a financial return or a treatment effect. Keep the non-significant result next to the number.

**Try it**

Does a positive forecast gain establish that supporting the predictor will reduce downstream stress? **Answer:** No. Prediction and intervention ask different questions, even when the predictor has a directed role.

## D12. Know which ideas remain interpretation

**Where we are**

Geometric interpretation and Limitations, pages 8–9; Methods and declarations, pages 9–11.

**What the passage is doing**

The paper separates its fitted-network results from a broader geometric motivation. A Markov blanket is a set of variables that screens a system from its surroundings in a specified statistical model. The proposed boundary-collapse story, learned geometry and related machinery are interpretive scaffolding here, supported chiefly by synthetic constructions and a limited empirical signature.

The draft also notes short series, uncertain rankings and the Asian-currency analysis on non-stationary levels. Horizon checks are reported for the studied networks; they are not a theorem for every possible model.

**In ordinary language**

A useful way to picture a problem is not the same as a tested law of nature. Check what was fitted, what was simulated and what was only proposed. A new domain needs its own data and tests.

**Evidence status**

Scope and limitations. The reported computations are CPU-only at moderate sizes, N ≤ 400. No massive-network GPU result or real intervention trial is established.

**Do not accidentally conclude**

The paper does not validate a universal geometric theory, physical flow map or emergency allocation policy. This guide does not recreate supplementary analyses absent from the selected PDF.

**Try it**

A sensor network has incomplete observations and unverified links. What transfers first? **Answer:** The questions and diagnostic workflow. A trustworthy source ranking or action benefit does not transfer without validation.

## Twelve questions to test transfer

Write answers before consulting the key.

### Core questions

1. How can the loudest node differ from the transmitter?
2. What does the pipeline fit before it tests support allocation?
3. What changes in the symmetrization test, and why is the growth setting controlled?
4. Why can a static source-aware rule be a serious benchmark for a dynamic planner?

### Boundary questions

5. Are the reported cascade-reduction percentages measured effects of real interventions?
6. Were the original five prediction-first tests externally preregistered before their analyses?
7. Does the positive 3.3% raw-data forecast result provide statistically significant confirmation?
8. Does fitted forecast direction establish the causal effect of supporting a node?

### Transfer questions

9. A highly connected new network is nearly symmetric. What would you test before expecting a targeting advantage?
10. A rolling analysis changes its leading source. How would you compare a fixed and an adaptive rule without future information?
11. A positive controller advantage has a wide interval spanning zero. How should a briefing describe it?
12. A nearest-neighbor map is offered as the physical network. What must be verified before treating its arrows as intervention targets?

### Answer key and reasoning

1. The loudest node has the greatest chosen stress summary; the transmitter has the largest positive net forecast-variance export.
2. A calibrated, non-negative VAR surrogate fitted to empirical stress series. Support is then simulated on a rescaled model.
3. The directional matrix component is attenuated. Rescaling each matrix limits confounding by a changing amplification regime in the simulation.
4. A fixed source can capture much of the gain when the useful directed structure is stable. Complexity must beat that baseline fairly.
5. No. They are calibrated-surrogate outcomes with model, budget and rescaling assumptions.
6. No. The draft distinguishes these records from three later externally timestamped prospective predictions not analyzed here.
7. No. The draft reports p = 0.50 and calls the forecast result suggestive.
8. No. Predictive direction is distinct from interventional causation.
9. Diagnose asymmetry and the transmitter–loudest gap, compare against strong baselines, and repeat the symmetrization test with uncertainty.
10. Use successive past-only windows, the same evaluation protocol and a clearly separated idealized oracle comparison.
11. A positive but imprecise model-based estimate. Do not omit the interval or present a confirmed field benefit.
12. Verify link meaning, source coverage, timing, model adequacy and the intervention mechanism. Proximity alone establishes none of these.

### One-sentence teach-back

Explain the core result to a colleague using four elements: the targeting comparison, its model-based evidence, its main causal/field limitation, and what a new domain must verify.

## Exact source

This companion is tied to the exact 5 July 2026 draft linked above. Its SHA-256 is:

b48f19b5a71943f540728fb197cb5d84aeea782a2aaad5a8b6144fe216755509

The manuscript's own reference list, pages 12–13, provides the cited methods and prior work. This teaching text adds no new empirical results. The export/import arithmetic in D2 is explicitly hypothetical. Numerical findings elsewhere are reported as findings of the selected draft, not newly rerun analyses.
