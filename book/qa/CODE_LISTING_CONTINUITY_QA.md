# Code Listing Continuity and Float-Placement QA

## Hard rule
No figure, table, sidebar, callout, algorithm float, or other floating object may appear between two portions of the same code listing.

## Required checks
1. Identify every listing that crosses a page boundary in the compiled PDF.
2. Verify that continuation code begins immediately on the next page, before any figure/table/float.
3. Verify continuous code order and, where used, continuous line numbering.
4. Verify that the listing retains one title/caption and one stable label.
5. Reject any manuscript construction that closes a listing, inserts a float, and reopens the same logical listing.
6. Verify nearby figures and tables are placed entirely before the listing or entirely after it.
7. Where float pressure causes interruption risk, move the float, move the listing, shorten the printed listing, or move nonessential code to GitHub.

## LaTeX production guidance
- Prefer non-floating listing environments for substantive printed code.
- Keep figures/tables from entering listing continuation space through appropriate float barriers or placement controls.
- Do not use manual page composition that changes code order.
- Final acceptance is based on the compiled PDF, not only the source ordering.

## Release criterion
- Interrupted code listings: 0
- Multi-page listing continuity failures: 0
- Split logical listings created solely to accommodate floats: 0
- Unresolved listing/float placement defects: 0
