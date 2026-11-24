# open-edit: authored design and render (DESIGN + RENDER variants B, REMIX and NO VIDEO)

Read this whole file before any authoring touch. The routing between variants and the compiled-recipe
variant A are in `SKILL.md` under DESIGN + RENDER.

## Re-read before every authoring touch

**On the authored variants (B, REMIX, NO VIDEO), re-read this file before you author anything, and
again before every authoring touch after it.** The whole file, not a section. It binds by what you do, not by the variant: a recipe run that reaches CUSTOMISING or a
classic PARAMETER AMEND is an authoring touch; only a recipe run that touches nothing reloads nothing.

**Reloading is not permission to redesign.** The system is committed once and not reopened; a later
touch places, sizes and times what the system decided. An authoring touch is one where you chose to
change how something reads.
**A mechanical fix to the element a gate flagged is not an authoring touch**: step the rung down,
close the window, move on, no reload.

Re-read it at each of these, every time:

- before writing `design/system.json`;
- before authoring any `.wv` document, and before each chapter of a longer piece;
- before adding or restyling any caption, graphic, plate, chart, mark, title, end card or motion,
  however small the touch.

A re-read replays no step this run has already finished:

- a recorded transcription provider is used and nothing is asked;
- the style draw is deterministic on the run key;
- preflight is idempotent;
- if `runs/<key>` already holds the footage, the FOOTAGE step is done and none of its gates are
  re-entered.

## Contract B and the gates

**B. FROM-SCRATCH (creative face-1)**: the base design contract driven by the user's materials (reference
video/images/brand kit/described concept), with 1-2 nearest recipe sheets from the runtime index as
engine-proven craft substrate. Execute it inline yourself per CREATIVE PASS (no subagent): read
`pipeline/director-brief.md`, study the materials, author and gate the .wv document in this session.
Commit one design system in one pass (no aesthetic re-litigation), author one single-timeline `.wv`
over the full footage, self-verify with lint + `veed-engine-cli --verify`, then render. Raw prefabs are not inputs. Refine re-runs after
ANALYSE use this same variant; ANALYSE says how to fill the materials slot.

Execution contract (follow it yourself, filling {...}):
```
Author one captioned composition over the footage as a single-timeline .wv document, verify it with
--verify, then render it.
CONTRACT (obey exactly): read {content}/pipeline/director-brief.md, the full engine contract (paint order,
the opacity/stacking trap, the one safe reveal recipe, engine limits, single timeline, render+verify).
INPUTS:
  - {repo}/runs/{key}/meta.json: canvas W/H/fps + durationSec + paths (authoritative for the manifest).
  - {repo}/runs/{key}/analysis.json, only if it exists (refine runs): per-beat facts in canvas px (shot,
    subjectBbox, faceBbox, negSpaceRect, brightness, caption text + start/end/mid/midFrame). If present,
    compose from these numbers; if absent (the default), from the transcript + the brief's safe margins.
    Either way, never read the frames.
  - {repo}/runs/{key}/transcript.json: captions + windows; chunks carry real per-word timings in
    `words: [{text, timestamp:[start,end]}]`. Animate each word on its own window; even-split only if a
    chunk has no `words` array.
  - video {videoPath} (in meta.json).
METHOD: the USER MATERIALS below are the design authority. Open them (images/files included) and study
their design DNA: positions, fonts, weights, sizes, colours, letter-spacing, shadows, mood. Translate it
into an engine-safe system; adopt and remix, never copy their content verbatim. From the CRAFT SUBSTRATE
sheets lift mechanics (timing idioms, grounding, width budgets, engine workarounds), not their look,
unless the user's materials point the same way.
USER MATERIALS (design authority): {paths / links / the user's described concept}.
CRAFT SUBSTRATE (recipe sheets, engine-proven; nearest by facets):
  1. {content}/refs/html/{idA}/recipe.md   2. {content}/refs/html/{idB}/recipe.md
DIRECTION = {content angle + mood + placement intent; no fonts/palette/device}.
ENGAGEMENT MODE = {seed copy, verbatim}.
ANIMATION LEVEL = {word | cue | none}.
TASK: write the system down first, then author against it.
  npx @veedstudio/openedit-cli design-gate {repo}/runs/{key}
  fails until {repo}/runs/{key}/design/system.json exists and every document obeys it. Author that file
  before any .wv: 2-3 Google @import fonts, a type ladder (each rung a role + size + its own optical
  tracking; `opticalTracking(px)` in {content}/pipeline/recipes/type.ts gives the measured curve), a named
  palette, spacing, named easings and durations, the reveal unit, the devices in play, and `donors` = the
  recipe ids you took mechanics from (checked against the runtime index). `groundedIn` must name the
  run's own content files; the gate refuses a system whose files do not exist.
Learn the repertoire first: open two or three ref folders under {content}/refs/html/ (sheet and document)
  and read what the engine is shown doing: a word set at a different size from its neighbours (a beat is
  a column of separate text blocks, not one styled line), underlines, arrows, brackets, corner marks,
  rules and badges made well, which faces are already proven to render. Write down what you found and
  what you intend to use. A bank of what is possible, not a template: look at how something is done,
  then do it better for the piece in hand.
COMPOSE, don't type: {content}/pipeline/recipes/devices.ts (dividers, ground shadow; a rule is a hairline +
  shadow + stub that draws, never a lone grey line), {content}/pipeline/design/captions.ts (per-word reveal
  off the real timings, travelling cursor, lines as blocks), {content}/pipeline/recipes/geometry.ts (arcs,
  lattices, springs, clip polygons), {content}/pipeline/recipes/type.ts (ladder, tracking curve). Contrast
  over footage comes from the two-layer ground shadow, not a scrim box.
Placement is a per-beat decision, written down: for every beat say where the block sits, why (what is
  behind it, which way the subject faces, where the frame is empty) and what changed since the last one.
  One position held for the whole piece is a defect. Emphasis is the word that takes the beat, not a
  phrase in bolder type. With footage, the system's `placement.measuredIn` must name a file with one
  entry per cue measured off this run's own picture; the gate refuses the system otherwise.
Not across the face: type passes behind a person, never across their eyes. When the empty ground runs
  out, use a smaller rung, a re-broken line, a different anchor or a later window, never the face. Read
  the face box off the subject's own silhouette (the head is the top of its bounding box) and keep it
  clear.
Graphics go where the content asks, not everywhere and never as decoration: a number spoken in the
  script invites a count, bar, scale or comparison; so does anything a picture states faster than the
  sentence. Whether to use them at all: if the reference carries graphics between the talking, that is
  the permission and the measure of how much; if the user's ask implies them, the same. Where neither
  does, don't.
A device that appears twice needs a logic: it follows a rule the piece keeps (every subject turn, every
  figure, every change of speaker) or it does not appear at all. A recurring device varies: same logic,
  different corner, different scale.
Lines are separate blocks, so give them different edges: lines all starting at the same x are a
  paragraph, not a composition. An indent, a hang or a step is free; use it, and let the size change
  inside the block so one word carries the beat.
Hold the system across all beats; vary scale/composition per beat; escalate hook to close. Author one
{repo}/runs/{key}/final/template.wv (z0 base video first; every text layer position:absolute + explicit
z-index>=1 + a unique `id` e.g. id="cap3", see the opacity trap; the id makes --verify name the element
in its failure lines; each caption visible only in its cue window) +
{repo}/runs/{key}/final/manifest.json {"render":{"width":W,"height":H,"fps":FPS,"duration":durationSec}}.
RENDER + VERIFY (outside any sandbox; {engine} = the renderer preflight installed, printed by
`npx @veedstudio/openedit-cli engine-path`: app-data on both platforms, moved by `VEED_ENGINE_BIN` or
`OPENEDIT_STATE_DIR`, not on PATH, so ask for it rather than typing it):
  DESIGN GATE (mechanical, no engine), run first: npx @veedstudio/openedit-cli design-gate {repo}/runs/{key}
     Reads every .wv in the run against design/system.json: an undeclared font, size, tracking, colour
     or easing is an error, as is a donor id that is not a real ref. Exit 1: fix the document, or amend
     the system deliberately.
  LINT (mechanical, no engine): npx @veedstudio/openedit-cli lint {repo}/runs/{key}/final/template.wv
     Engine-limit anti-patterns (animated blur, the stacking trap, missing cue ids, per-corner radius).
     Exit 1: fix the flagged rule, re-lint before verifying.
  VERIFY (analytic, fast, no video, reads manifest render block):
       {engine} {repo}/runs/{key}/final --verify=bounds,safezones --verify-report {repo}/runs/{key}/final/verify.json
     Replays the whole timeline offscreen against the real draw list. Exit 0 = no problems. Exit 1 = one
     stdout line per problem naming the element id, e.g.:
       frame 3 t=0.400s FAIL[bounds] #cap3 glyph 14 right 3.1px outside (8.42% of glyph box) viewport 736x1312
       FAIL[never-visible] #cap5 glyph 2 ink in 300 frames, never fully visible (best 0.00% at frame 0 ...)
       frame 2 t=0.200s FAIL[occluded] #cap2 glyph 5 fully covered by later opaque rect
       FAIL[safezone] #b4w31 zone generic-9x16 53.5% of ink outside keep-inside, worst frame 314 t=10.5s, window 10.5..11.3s, 24 offending frames (longest run 24)
     Rules: bounds (type off the viewport), never-visible (type clipped away in every frame, e.g. behind
     a mask/box), occluded (type fully hidden under a later opaque layer, the z-order/opacity trap),
     safezone (type inside the viewport but outside the platform-safe area; SAFE-ZONE CHECK owns its
     triage and fix). Fix only the flagged element (nudge inside the safe zone / fix z-order or the mask)
     and re-run --verify until exit 0. Exit 2 = engine render failure = a real authoring error. If the
     engine prints its usage instead of a report, it predates `--verify=<rules>` (`--help` lists
     `--verify[=<rules>]` on an engine that has it): run bare `--verify` for the first three rules and
     say plainly that the safe-zone check did not run; never fake it from a screenshot.
  SAFE-ZONE CHECK: the `safezones` family of the VERIFY invocation above. Same walk, no extra cost; it
     judges glyph ink only (plates, boxes, images, video and semi-transparent fills are not observed)
     against the platform-safe area; the check and every fix are the document's, the mp4 is never
     inspected.
     Zones: bare `safezones` picks the generic preset by canvas aspect: 9:16 = x 6..89% / y 11..83% (top
     11%, bottom 17%, left 6%, right 11%: the brief's own safe margins, the band feed UI covers); 16:9 =
     6% inset; 1:1 = 5% inset. Custom zones when the user's
     platform or brand says so: `--verify=bounds,safezones:{path}.json` with
     `{"zones":[{"name":"...","rect":{"x":%,"y":%,"w":%,"h":%},"mode":"keep-inside|keep-out","severity":"error|warn"}],"exempt":["id"]}`
     or the same object as `"verify":{"safezones":{...}}` in manifest.json, which wins over the flag.
     `keep-out` = a rect ink must stay out of (a logo corner, a sticker band); `warn` reports and scores
     but exits 0; `exempt` = ids whose bleed is the design (a full-bleed title, a ticker), never an id
     you want to stop failing.
     Read the report: one FAIL[safezone] line per (element, zone), aggregated over the timeline. `% of
     ink` = intruding ink-area over visible ink-area across every frame; `worst frame` = where to look;
     `window` = first..last offending time; `N offending frames (longest run M)` = how long the ink was
     actually out (the window alone cannot tell one flash from two far apart). The element named is the
     innermost id'd element carrying the ink, here a word span (`#b4w31`); move the line or block that
     positions it (its `.ln` / page container); the words of one line move together. `verify.json` adds
     `max_intrusion_px` per violation (depth in canvas px; size the fix from it), `alpha_at_max_intrusion`
     (effective opacity in that deepest frame; a slide-in still fading up posts a deep number nobody
     sees), `offending_frames` / `longest_run_frames`, per-element scores and an `overall_score` (100 = no
     intrusion; ink-area-time weighted; describe with it, never triage). To see a worst frame:
     `{engine} {repo}/runs/{key}/final --headless --frame-num-until-exit {frame} --exit-screenshot {path}.png`.
     Chrome first: an id ending in `-chrome` is dressing (kickers, credits, film-strip labels and marks,
     stickers), not the spoken line. Put the suffix on the element that directly wraps the text (the
     engine labels a run by its direct parent's id; an id one level up leaves the run reported by its
     text, pooled with every other run of the same text). A `FAIL[safezone]` on a `-chrome` id is not triaged
     and never fixed: name it in one clause at delivery ("the credits strip runs under the feed UI") and
     deliver. Only content violations enter the triage.
     Triage: TRANSIENT vs MINOR vs MAJOR, per positioned line, from the numbers, never by eye. `edge` =
     2% of the canvas's shorter side (22px on a 1080-wide canvas); `hold` = 250ms in frames (8 at 30fps,
     6 at 25). A report without `longest_run_frames` (older engine) uses the `window` length as the hold;
     it overstates two flashes far apart, never understates a held line.
       TRANSIENT: every violation on the line has `longest_run_frames` < hold and `ink_intruding_fraction`
         <= 0.05 (an entry or settle crossing the margin; `max_intrusion_px` is the animation's travel,
         not a placement). Not a fix: deliver and mention it in one clause ("two lines cross the safe
         margin for a few frames on entry"). Call it a visible flick only when `alpha_at_max_intrusion`
         is above 0.5.
       MINOR: not transient, and every violation on the line has `max_intrusion_px` <= edge and
         `ink_intruding_fraction` <= 0.25 (a descender, an overshoot, a line box set on the margin).
         Autocorrect without asking: shift that line's container
         inward by ceil(max_intrusion_px) + 4px (the side the line sits nearest; the document tells you,
         not the report), re-run VERIFY, and mention it in one clause at delivery ("nudged two lines
         inside the safe area").
       MAJOR: anything else (deeper than `edge` and not transient, more than a quarter of the ink out, or
         `100%`, the whole element outside; a band under the feed UI, a title wider than the zone, a
         wrong-canvas assumption). Autocorrect as a re-placement, not a nudge: put the block at
         the nearest legal position on its own side (bottom band: top = zone bottom px - block height -
         8px; a side: left/right = zone edge + 8px); if the block is wider than the zone (left and right
         both fail, or its width exceeds the zone's) step it one rung down the size ladder first, then
         place; if `analysis.json` exists and the new position lands on `faceBbox`, take the other band.
         Re-run VERIFY. Tell the user what moved and by how much in plain terms, not the rule name. Never
         resize, recolour or re-time anything else.
       keep-out zones triage on the same numbers (depth = how far into the rect, fraction = how much ink
         is in it, hold = how long); the fix moves the ink out the shortest way.
       `WARN[safezone]` lines never block; treat as minor for the message, fix only if the fix is a nudge.
     At most two correction cycles, then stop and report. On a compiled-recipe run the document is
     script-owned and a safezone FAIL is a recipe geometry bug: chrome, delivered and mentioned;
     TRANSIENT, delivered and mentioned (entry animation, not placement); MINOR, delivered and reported
     (the recipe's own margins are the fix, offline);
     MAJOR, fixed on a `--module` copy (the CUSTOMISING route), never in `final/template.wv` by hand.
     If asked to check a delivered result ("is this inside the safe zones?"), run the same command on
     its run dir's `final/`; an mp4 with no run dir has nothing to replay, so say so rather than judging
     frames by eye.
  EXPECT WINDOWS (optional), word-reveal timing, to assert a caption is shown/hidden in a time window: add
     a "verify" block to manifest.json alongside "render":
     {"verify":{"expect":[{"element":"cap3","visible":true,"from":2.1,"to":3.4}]}}. --verify then
     FAILs[expect-visible]/[expect-hidden] if a word isn't on-screen when it should be. Use when a beat's
     reveal timing is subtle; ids only (a word with no id can't be targeted).
  WCAG PASS (DEFAULT on creative runs; level AA) — after --verify is clean, before recording. It
     DETECTS and REPORTS; it NEVER silently changes colours — the human chooses.
       npx @veedstudio/openedit-cli wcag-pass --run {repo}/runs/{key}
     It samples the REAL rendered background behind every caption, checks WCAG AA contrast, writes
     final/contrast-statistics.json (the policy-free statistics the verdicts and the proposals are
     computed from), and STOPS. It prints `status: pass|attention` plus ONE `propose:` line per failing
     CLASS. Exit 1 = tooling missing or crashed (fix the environment: the engine carries the analyzer,
     so install or update it with npx @veedstudio/openedit-cli install-engine) — not a design failure.
     ONE REMEDIATION PER CLASS: every text of a class takes the SAME fix, solved at the class's common
     denominator, so a class is the unit the user chooses at. Each line offers up to three rungs,
     RECOMMENDED FIRST — `colour` (only when one colour covers the whole class), `shadow` (a solved
     recipe, not a guess: either a SOFT stack printed as `x{layers} @ {blur}px`, or a HARD faux-outline
     ring printed as `{directions}-way @ {offset}px` when no soft stack reaches AA), `background` (text
     anchored to black/white on a solid plate). When no colour covers the class the line SAYS SO and
     offers only shadow and background; when nothing reaches AA it says that too. Say `background` to the user as "a solid box" — the rung NAMES are the choice file's
     own values, so copy them across verbatim and never translate one into another word.
     - `status: pass` → all caption text clears AA. Note it and move on.
     - `status: attention` with the note `0 runs audited` → the audit examined nothing: the statistics
       file names no text elements. Say so plainly and offer no fixes — there is no class to choose at.
     - `status: attention` → some caption text is below AA. Speak it in PLAIN PRODUCT TERMS, briefly,
       for someone who does not know what a CSS class is. Head it `Text contrast audit — target rating:
       WCAG AA`, give the count, then ONE line per class: the class label (the `propose:` selector with
       its `.`/`#` dropped, `-`/`_`/`.` turned into spaces, sentence-cased — `.caption-emph` reads as
       "Caption emph"), the recommended rung, and the alternatives in passing. Then close on exactly
       THREE choices, and do NOT pick for them:
         Apply    make these changes
         Ignore   record the video unchanged
         Adjust   name a different option
       - APPLY or ADJUST → WRITE runs/{key}/final/wcag-choice.json yourself from what they said
         ({"schema":1,"chosen":[{level:"AA",selector,kind:"colour"|"shadow"|"background",hex,backingHex?,recipe?}]};
         `selector` is the class key from the `propose:` line COPIED VERBATIM, one entry per class,
         `backingHex` required for `background`, `recipe` carried VERBATIM from the offered shadow —
         `{style:"soft",layers,blur}` or `{style:"hard",directions,offset}`, never converted between the
         two), then run the
         apply step. A selector naming no text in the template ABORTS the apply step and lists the valid
         keys — so copy them, never invent or reformat one. On ADJUST keep the classes they did not
         mention on their recommended rung.
       - IGNORE → record the original unchanged; write nothing.
     - APPLY STEP — a chosen option ALWAYS takes effect:
         npx @veedstudio/openedit-cli wcag-pass --run {repo}/runs/{key} --apply
       With a runs/{key}/final/wcag-choice.json present, --apply drives the applier from that explicit
       choice (colour / shadow / background box) and ALWAYS runs it, promoting unconditionally; with NO
       choice file it falls back to the automatic hue-preserving colour set (value/saturation shift only,
       design identity kept), applied only where a colours-only fix covers a whole class and promoted only
       on MEASURED improvement. FINAL IS FINAL — no sibling dirs,
       no video artifacts; on promotion the remediated template becomes runs/{key}/final/template.wv
       and the file artifacts land inside final/: template.draft.wv (pre-remediation original),
       template.draft.wcag-remediated.wv (the remediation output), template.final.wv (clone of
       the shipping template on promotion), wcag-remediation.css (per-rule evidence),
       wcag-choice.resolved.json (the chosen class keys resolved to the element ids actually styled),
       wcag-remediation-plan.json, contrast-statistics.json. SAMPLING HAPPENS ONCE, in the detect run:
       a decoration is painted OVER the footage, so its effect is COMPUTED against those same samples
       and the remediated render is never re-sampled. After promoting it re-runs
       --verify on final/ itself, and prints `status: pass|remediated|not-improved|residual` + the
       same per-class `review:` lines.
  RECORD the deliverable — ONLY after --verify is clean, always FROM runs/{key}/final (whatever the
     user chose in the WCAG PASS already lives in final/template.wv — the untouched original, or an
     --apply promotion). --verify and --record are mutually exclusive, so this is a
     SECOND invocation:
       {engine} {repo}/runs/{key}/final --progress-output --record {repo}/runs/{key}/final/out.silent.mp4
     --progress-output prints `progress: N/M frames (X%)` lines during the record; recording is long-running —
     watch these to confirm it's alive, but don't narrate them to the user.
  Change nothing else: no aesthetic/colour/font/device/animation/timing edits; --verify is a safety net,
  not a design loop.
  probe-qa (mechanical frame QA) comes right after the record step, not inside the verify loop.
OUTPUT: {repo}/runs/{key}/final/{template.wv, manifest.json, out.silent.mp4}.
THEN (same turn, no pause): note the locked system so you can describe the delivered look in plain terms
(aesthetic, fonts, palette, device, not gate status), note the wcag status (pass / attention with what
the user chose: Apply, Ignore, or an Adjust), and continue straight to probe-qa + mux.
```

After recording, run the last gate: `npx @veedstudio/openedit-cli probe-qa runs/<key>` (mechanical
frame QA: per-beat mid + tail probes vs the source, caption present, WCAG-ish contrast).
- FAIL: do not redesign and do not auto-re-render. Report it honestly in plain terms and pick the
  fix with the user.
- Warns: proceed to mux; surface one only if it is likely visible in the deliverable.

**REMIX EXECUTION (face-2, inline — no subagent).**
- Make a fresh run dir `runs/<key>-remix`: copy `meta.json`, `transcript.json`, `word-timings.json`
  from `runs/<key>` and set the `key` field in `meta.json` to `<key>-remix`.
- Pick donors per SAMPLE ONE STYLE: A = the current pick's sheet as skeleton, same aspect; B = a
  different `type` facet; optional C = a different `device` facet.
- Write `runs/<key>-remix/design/system.json` first, as on face-1: copy the original's, change what
  the brief asks to change, name the donors in `donors`.
- Author `runs/<key>-remix/final/{template.wv, manifest.json}` per `director-brief.md` REMIX MODE.
- Run the SAFE-ZONE CHECK (the VERIFY invocation in the RENDER + VERIFY block) on
  `runs/<key>-remix/final` before the chain; the chain's `--verify` is the bounds family only.
- Run the chain (outside any sandbox): `npx @veedstudio/openedit-cli gates runs/<key>-remix`
  It runs design → lint → `--verify` → WCAG → `--record` → probe-qa → mux, stops at the first failure
  and names the gate.
- `--verify` failure: fix only the flagged element and re-run, at most twice, then stop and report
  honestly.
- probe-qa failure: the probe-qa policy above applies.
- The deliverable lands next to the original; the user compares.
- The WCAG AA pass runs inside the chain, before the record; do not run it again afterwards. It detects
  and reports only. On `status: attention` follow the WCAG PASS block in full. On Apply or Adjust
  write `runs/<key>-remix/final/wcag-choice.json` yourself from what they said and run
  `… wcag-pass --run runs/<key>-remix --apply`; on Ignore leave it as-is. Never pick for them. After
  an apply re-run the chain so the record is made from the promoted template.

## MUX AUDIO for a built soundtrack

If the audio is built rather than restored (narration, music, effects, anything with more than one
piece), write `runs/<key>/audio/mix.json`, build the track, and mux that:
```json
{ "durationSec": 726.8, "tracks": [
  { "path": "assets/vo-01.mp3",  "atSec": 0,    "role": "voice" },
  { "path": "assets/music-1.mp3","atSec": 12.4, "gainDb": -14, "fadeOutSec": 3, "role": "music", "duck": true },
  { "path": "assets/sfx-3.mp3",  "atSec": 88.2, "gainDb": -6,  "role": "sfx" } ] }
```
```
npx @veedstudio/openedit-cli mix-audio runs/<key>          # → runs/<key>/audio/mix.m4a
npx @veedstudio/openedit-cli mux-audio runs/<key> --audio runs/<key>/audio/mix.m4a
```
- The mix is levelled as a whole at mux; set the pieces' relative gains in `mix.json` and leave the
  overall level to it.
- `durationSec` is required and is the film's length; anything past it is trimmed.
- A bed marked `duck` is opened by the voice itself, not by a guessed gain.
- A fade-out is measured from the end of the film.
- `--print-graph` shows the filtergraph without running ffmpeg.

## Engagement modes — keep the seed copy VERBATIM (wording changes output)
- **scroll-stopping** (default, social): "the first frame must STOP the thumb — big numbers, oversized, chrome/neon, saturated; optimize for a feed."
- **wow**: "push the WOW ceiling — oversized, bleed past edges, heavy effects."
- **design-grade**: "gallery-quality AND maximally engaging — high-design, premium, crisp."
- **variety / bold-broadcast** (good for 16:9 landscape): "loud broadcast / sports-lower-third energy; big type in the landscape thirds; heavy effects."
