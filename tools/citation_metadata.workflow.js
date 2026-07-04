/**
 * Adversarial citation-metadata check (resolve -> refute) — the LLM complement to the
 * deterministic tools/check_citations.py.
 *
 * WHY BOTH: the deterministic gate scores by title containment, so it PASSES a real paper
 * that carries the wrong volume/issue/pages/venue (a right-title/wrong-year case is now caught
 * deterministically too, but the finer metadata fields are not). An adversarial pass resolves
 * each reference against the authoritative record and refutes any field mismatch. Neither alone
 * suffices — this is the "run both" lesson from the false-floor sibling run.
 *
 * Input (args): { entries: [{key, text, doi, arxiv, year}], ... }  — produced deterministically by
 *   python3 tools/check_citations.py <manuscript>.tex --dump-entries
 * so the (tested) parser, not an LLM, defines the reference list.
 *
 * Pipeline per reference: RESOLVE (web -> authoritative record) then an INDEPENDENT REFUTE
 * (field-by-field diff). Returns only genuine mismatches + anything unresolvable.
 */
export const meta = {
  name: 'citation-metadata',
  description: 'Adversarial resolve->refute check of every reference\'s metadata (title/authors/year/venue/volume/issue/pages)',
  phases: [
    { title: 'Resolve', detail: 'find the authoritative record on the web' },
    { title: 'Refute', detail: 'independent field-by-field diff vs the citation' },
  ],
};

let _A = args;
if (typeof _A === 'string') { try { _A = JSON.parse(_A); } catch (e) { _A = null; } }
const ENTRIES = (_A && _A.entries) || [];
log(`args type=${typeof args}; resolved ${ENTRIES.length} references`);

const RECORD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['found', 'confidence'],
  properties: {
    found: { type: 'boolean', description: 'true if an authoritative record was located' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    title: { type: 'string' }, authors: { type: 'string' }, year: { type: 'string' },
    venue: { type: 'string', description: 'journal / conference / publisher' },
    volume: { type: 'string' }, issue: { type: 'string' }, pages: { type: 'string' },
    doi: { type: 'string' }, sourceUrl: { type: 'string' },
    note: { type: 'string' },
  },
};

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['status', 'mismatchedFields'],
  properties: {
    status: { type: 'string', enum: ['clean', 'mismatch', 'unresolved'] },
    mismatchedFields: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['field', 'cited', 'record'],
        properties: {
          field: { type: 'string', enum: ['title', 'authors', 'year', 'venue', 'volume', 'issue', 'pages', 'doi'] },
          cited: { type: 'string' }, record: { type: 'string' },
          severity: { type: 'string', enum: ['minor', 'major'] },
        },
      },
    },
    note: { type: 'string' },
  },
};

const results = await pipeline(
  ENTRIES,
  (e) => agent(
    `Find the AUTHORITATIVE bibliographic record for the reference cited below. Use live web search ` +
    `(CrossRef, the publisher's page, arXiv, Google Scholar). Prefer the version of record. Return its ` +
    `metadata fields exactly as the authority lists them; set found=false if you cannot locate a matching ` +
    `record. Do NOT copy the cited string back — resolve it independently.\n\n` +
    `CITED: ${e.text}${e.doi ? `\nDOI: ${e.doi}` : ''}${e.arxiv ? `\narXiv: ${e.arxiv}` : ''}`,
    { label: `resolve:${e.key}`, phase: 'Resolve', schema: RECORD_SCHEMA, effort: 'high' }
  ).then((record) => ({ e, record })),
  ({ e, record }) => {
    if (!record || !record.found) {
      return { key: e.key, cited: e.text, status: 'unresolved',
               verdict: { status: 'unresolved', mismatchedFields: [], note: record?.note || 'no record located' } };
    }
    return agent(
      `You are a skeptical bibliographic fact-checker. Compare the CITED reference against the independently ` +
      `RESOLVED authoritative record, field by field: title, authors, year, venue, volume, issue, pages, doi. ` +
      `Report ONLY genuine mismatches (a wrong year, volume, issue, page range, venue, or author set). IGNORE ` +
      `pure formatting differences (abbreviations, initials vs full names, "&" vs "and", en-dash vs hyphen, ` +
      `journal-name abbreviation). If the record confidence is low or the two clearly describe different works, ` +
      `say status=unresolved rather than inventing a mismatch.\n\n` +
      `CITED: ${e.text}\n\nRESOLVED RECORD: ${JSON.stringify(record)}`,
      { label: `refute:${e.key}`, phase: 'Refute', schema: VERDICT_SCHEMA, effort: 'high' }
    ).then((verdict) => ({ key: e.key, cited: e.text, record, verdict }));
  }
);

const clean = results.filter(Boolean);
const mismatches = clean.filter((r) => r.verdict?.status === 'mismatch');
const unresolved = clean.filter((r) => r.verdict?.status === 'unresolved');

return {
  total: ENTRIES.length,
  clean: clean.length - mismatches.length - unresolved.length,
  mismatchCount: mismatches.length,
  unresolvedCount: unresolved.length,
  mismatches: mismatches.map((r) => ({
    key: r.key,
    fields: (r.verdict.mismatchedFields || []).map((m) => `${m.field}: cited "${m.cited}" vs record "${m.record}"${m.severity ? ` [${m.severity}]` : ''}`),
    note: r.verdict.note || '',
    source: r.record?.sourceUrl || '',
  })),
  unresolved: unresolved.map((r) => ({ key: r.key, note: r.verdict?.note || '' })),
};
