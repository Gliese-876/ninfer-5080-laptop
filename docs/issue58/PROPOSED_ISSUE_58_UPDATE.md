# Proposed Issue #58 update (draft — do not publish)

C2 temporal-credit investigation is closed as **rejected**, with the evidence preserved for future work.

The tested `refresh_temporal_credit` candidate overwrote valid temporal credit with zero in the exercised policy scenario: the original probe accepted temporal backfill (RC 0; credit 20), while the candidate rejected it (RC 1; credit 0). These are synthetic policy checks, not a reproduction of the original user-visible workload.

A source-preserving baseline review found no authoritative original failing request sequence in the retained evidence, so protected-head starvation is recorded as **not reproduced**, not disproven. Reopening this line requires either a concrete baseline trace that exceeds the donor-frontier bound or an approved replenishment invariant with an anti-delay proof.

Issue #58 remains open. Separate work remains for reversible KV/MTP lane yield, physical KV export/import, shared-executor lifecycle validation, and authorized GPU qualification.

Evidence is indexed in `docs/issue58/` for review; no scheduler code, service state, or production configuration changed in this archival update.
