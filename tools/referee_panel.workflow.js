/**
 * Upgraded adversarial referee panel (Tier-4 harness) — reusable Claude Code Workflow.
 *
 * Encodes the 2025-2026 best practices that make an AI review CATCH errors instead of
 * generating fluent praise:
 *   1. TARGETED error-hunt lenses — each agent hunts ONE named error class with an explicit
 *      definition (LLMs miss flaws under holistic "review this paper" prompts).
 *   2. Grounding — every finding must quote a VERBATIM passage; the caller then verifies the
 *      quote exists in the source deterministically (grep), dropping any hallucinated passage.
 *   3. Generate-then-REFUTE — each finding faces N skeptics prompted to refute it; only findings
 *      that survive a majority survive (kills plausible-but-wrong findings).
 *   4. Calibration control — a gutted, numberless stub is run through a lens; if the panel
 *      "finds" serious problems in an empty document, it is fabricating and its verdicts are void.
 *
 * Args (optional): { ms: "<path to manuscript.tex>", si: "<path to supplementary.tex>" }
 * Returns findings (with refute tallies) + the calibration result; the CALLER grounds the quotes.
 */
export const meta = {
  name: 'referee-panel',
  description: 'Upgraded adversarial referee panel: targeted error-hunt lenses, generate-then-refute, calibration control',
  phases: [
    { title: 'Error-hunt', detail: 'one named error class per lens' },
    { title: 'Refute', detail: '3 skeptics per finding; majority-refute kills it' },
    { title: 'Calibration', detail: 'gutted-stub control — must find nothing' },
  ],
};

const MS = (args && args.ms) || '/Users/patrickiben/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/manuscript.tex';
const SI = (args && args.si) || '/Users/patrickiben/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/supplementary_information.tex';

const READ = `Read the manuscript at ${MS} and the supplementary information at ${SI} in full before judging.`;

const LENSES = [
  { key: 'surrogate-only', errorClass: 'surrogate-only effect size',
    def: 'a headline effect size (e.g. an interdiction / cascade-reduction percentage) that is computed ONLY on the calibrated VAR(1) surrogate and never validated against a real out-of-sample outcome, yet is presented as if it were an empirical property of the real system.' },
  { key: 'null-leak', errorClass: 'null that leaks its own test statistic / collapses by construction',
    def: 'a null model whose construction guarantees the tested quantity vanishes (so "significance" is mechanical, not evidential) — inspect whether the symmetrization null could pass its verdict by construction and whether an independent null is offered.' },
  { key: 'oos-nonsig', errorClass: 'undisclosed or downplayed non-significant out-of-sample result',
    def: 'the held-out real-data out-of-sample test is non-significant (sign-test p=0.50), and the claim is that this must be stated plainly and NOT rhetorically upgraded to sound confirmatory.' },
  { key: 'circularity', errorClass: 'near-circularity among the core constructs',
    def: 'transmitter identity, connectedness, and interdiction advantage are all functions of the same fitted operator, so an apparent confirmation may be the same quantity measured twice — flag any place the paper treats correlated derivations as independent evidence.' },
  { key: 'overclaim', errorClass: 'unsupported superlative / universal quantifier',
    def: 'a false or unverified "most/least/every/always/no" claim (e.g. "the most connected network we examine", "every controller <= X%") contradicted by the paper\'s own numbers.' },
  { key: 'numeric-consistency', errorClass: 'internal numeric inconsistency',
    def: 'a number in the abstract, body, table, figure, or SI that disagrees with the same quantity stated elsewhere (counts of networks, confirm/refine/falsify tally, percentages, p-values, spectral radius).' },
  { key: 'citation-support', errorClass: 'claim not supported by its cited reference',
    def: 'a factual/method claim attributed to a citation that the cited work does not actually establish, or a citation used to support a stronger statement than it warrants.' },
  { key: 'stats-validity', errorClass: 'invalid or missing statistical control',
    def: 'multiple comparisons across the seven networks / many edges without correction; a significance claim without an uncertainty interval; a test whose assumptions (stationarity, independence) are unmet or unchecked.' },
];

const FINDING_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['errorClass', 'severity', 'claim', 'exactQuote', 'why', 'suggestedFix'],
        properties: {
          errorClass: { type: 'string' },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          claim: { type: 'string', description: 'one-sentence statement of the defect' },
          exactQuote: { type: 'string', description: 'VERBATIM substring copied from the manuscript/SI that the finding is about (<=200 chars). Must be copy-pasted exactly so it can be grep-verified.' },
          location: { type: 'string', description: 'section / nearby heading' },
          why: { type: 'string' },
          suggestedFix: { type: 'string' },
        },
      },
    },
  },
};

const REFUTE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['refuted', 'reason'],
  properties: {
    refuted: { type: 'boolean', description: 'true if the finding is wrong / not a real defect' },
    reason: { type: 'string' },
  },
};

phase('Error-hunt');
const perLens = await parallel(LENSES.map((L) => () =>
  agent(
    `${READ}\n\nYou are a hostile referee hunting for EXACTLY ONE class of defect and nothing else:\n` +
    `ERROR CLASS: ${L.errorClass}\nDEFINITION: ${L.def}\n\n` +
    `Report only real instances of THIS class. For each, copy the exact offending passage VERBATIM into exactQuote ` +
    `(so it can be verified against the source). Prefer at most the 5 most serious. If there are none, return an empty list — ` +
    `do NOT invent findings to seem thorough (a false finding is worse than none).`,
    { label: `hunt:${L.key}`, phase: 'Error-hunt', schema: FINDING_SCHEMA, effort: 'high' }
  ).then((r) => (r?.findings || []).map((f) => ({ ...f, lens: L.key })))
));

const findings = perLens.filter(Boolean).flat();
log(`error-hunt raised ${findings.length} candidate findings across ${LENSES.length} lenses`);

phase('Refute');
const judged = await parallel(findings.map((f) => () =>
  parallel([0, 1, 2].map((i) => () =>
    agent(
      `You are skeptic #${i + 1}. A referee raised this finding about the manuscript at ${MS} (+ SI ${SI}). ` +
      `Your job is to REFUTE it if you honestly can. Read the surrounding context. Default to refuted=true only if the ` +
      `finding is genuinely wrong, already addressed by the paper, or not a real defect; set refuted=false if the finding stands.\n\n` +
      `FINDING (${f.severity}, ${f.errorClass}): ${f.claim}\nQUOTE: "${f.exactQuote}"\nWHY: ${f.why}`,
      { label: `refute:${f.lens}`, phase: 'Refute', schema: REFUTE_SCHEMA, effort: 'high' }
    )
  )).then((votes) => {
    const v = votes.filter(Boolean);
    const refuted = v.filter((x) => x.refuted).length;
    return { ...f, refuteVotes: refuted, refuteTotal: v.length, survives: refuted < 2,
             refuteReasons: v.map((x) => x.reason) };
  })
));

const survivors = judged.filter(Boolean).filter((f) => f.survives);
log(`${survivors.length} of ${findings.length} findings survived majority adversarial refutation`);

phase('Calibration');
const calib = await agent(
  `CALIBRATION CONTROL. Below is a COMPLETE manuscript. Hunt for "${LENSES[0].errorClass}" ` +
  `(${LENSES[0].def}). Report every instance you find, verbatim quote required.\n\n` +
  `=== MANUSCRIPT ===\nTitle: A note on directed contagion.\nAbstract: We discuss directed contagion in networks ` +
  `and argue that direction can matter for intervention. We present a framework and outline avenues for future work. ` +
  `No empirical results are reported here.\nSection 1: Networks can be directed. Intervention may benefit from direction.\n` +
  `=== END ===\n\nIf this stub contains no instance of the error class, return an empty list. Returning any finding here ` +
  `means the panel fabricates defects in empty text.`,
  { label: 'calibration:gutted-stub', phase: 'Calibration', schema: FINDING_SCHEMA, effort: 'high' }
);
const fabricated = (calib?.findings || []).length;
log(`calibration: panel reported ${fabricated} findings in a numberless stub (want 0)`);

return {
  manuscript: MS,
  supplementary: SI,
  counts: { candidates: findings.length, survivors: survivors.length },
  calibration: { fabricatedInGuttedStub: fabricated, trustworthy: fabricated === 0 },
  survivors: survivors
    .sort((a, b) => (a.severity < b.severity ? -1 : 1))
    .map((f) => ({ severity: f.severity, lens: f.lens, errorClass: f.errorClass, claim: f.claim,
                   exactQuote: f.exactQuote, location: f.location, why: f.why, suggestedFix: f.suggestedFix,
                   refute: `${f.refuteVotes}/${f.refuteTotal} skeptics refuted` })),
  refutedOut: judged.filter(Boolean).filter((f) => !f.survives)
    .map((f) => ({ claim: f.claim, refute: `${f.refuteVotes}/${f.refuteTotal}`, reasons: f.refuteReasons })),
};
