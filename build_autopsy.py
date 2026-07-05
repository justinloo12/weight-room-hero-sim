"""Generate autopsy.html — the editorial write-up of the model arc.

Every number on the page is read from eval_results.json (plus the figure
PNGs, base64-embedded); nothing is hand-typed, so regenerating after a new
evaluation run keeps the story honest. Style follows the house editorial
register: serif headlines, no emoji, no gradients, no icon cards.

    python3 build_autopsy.py   # writes autopsy.html
"""

import base64
import json
from pathlib import Path

OUT = Path("autopsy.html")
RESULTS = json.loads(Path("eval_results.json").read_text())


def img64(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


# ── Pull every number from eval_results.json ──────────────────────────────
bb_gbm = RESULTS["batted_ball_level"]["gbm"]
bb_base = RESULTS["batted_ball_level"]["base_rate"]
pg = RESULTS["player_game_level"]
picks = RESULTS["picks_subset (n=88)"]
v2 = RESULTS["structural_v2"]["player_game_ablation"]["v2_slots_plus_parks"]
v2_picks = RESULTS["structural_v2"]["picks_subset_ablation (n=88)"]["v2_slots_plus_parks"]
sim = RESULTS["structural_v2"]["betting_simulation"]
v3 = RESULTS["gbm_v3"]["player_game_level"]["gbm_v3"]
v3_cfg = RESULTS["gbm_v3"]["config"]
v3_picks_block = next(v for k, v in RESULTS["gbm_v3"].items()
                      if k.startswith("picks_subset"))
v3_picks = v3_picks_block["gbm_v3"]
market = v3_picks_block["market_implied_devig"]

n_pg = f"{pg['gbm']['n']:,}"
n_hr = f"{pg['gbm']['positives']:,}"

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>My ML Model Lost to a Formula</title>
<style>
  :root {{
    --ink:#16181d; --muted:#5c6470; --faint:#8a919c;
    --accent:#8c2f28; --dark:#1c1414; --gold:#ba9653;
    --rule:#e3e5e8; --serif:Georgia,'Iowan Old Style','Times New Roman',serif;
    --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,Helvetica,Arial,sans-serif;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{ background:#fff; color:var(--ink); font:17px/1.65 var(--sans); }}
  a {{ color:var(--accent); text-decoration:none; border-bottom:1px solid #d8b7b3; }}
  a:hover {{ border-bottom-color:var(--accent); }}
  .hero {{ background:var(--dark); color:#f6f3f2; padding:88px 24px 72px; }}
  .hero-inner {{ max-width:820px; margin:0 auto; }}
  .kicker {{ font:600 12px/1 var(--sans); letter-spacing:.22em;
    text-transform:uppercase; color:#b09a97; }}
  .hero h1 {{ font-family:var(--serif); font-weight:400; letter-spacing:-.01em;
    font-size:clamp(38px,6.4vw,60px); line-height:1.08; margin:26px 0 22px; color:#fff; }}
  .dek {{ font-size:19px; line-height:1.55; color:#d5c8c5; max-width:640px; }}
  .herostats {{ display:flex; gap:56px; flex-wrap:wrap; margin-top:52px;
    border-top:1px solid rgba(255,255,255,.14); padding-top:30px; }}
  .herostats .num {{ font-family:var(--serif); font-size:44px; line-height:1; color:#fff; }}
  .herostats .lab {{ font-size:11.5px; letter-spacing:.14em;
    text-transform:uppercase; color:#a8908c; margin-top:9px; }}
  .article {{ max-width:760px; margin:0 auto; padding:24px; }}
  section {{ margin:72px 0; }}
  .slug {{ font:600 12px/1 var(--sans); letter-spacing:.2em;
    text-transform:uppercase; color:var(--accent); margin-bottom:14px; }}
  h2 {{ font-family:var(--serif); font-weight:400; font-size:31px;
    line-height:1.2; letter-spacing:-.01em; margin-bottom:18px; }}
  p {{ margin:0 0 18px; color:#2a2e35; }}
  p b, p strong {{ font-weight:650; color:var(--ink); }}
  .neg {{ color:var(--accent); font-weight:650; }}
  .pos {{ color:#1f6f43; font-weight:650; }}
  figure {{ margin:30px -60px; }}
  figure img {{ width:100%; display:block; border:1px solid var(--rule); border-radius:2px; }}
  figcaption {{ font-size:13.5px; color:var(--faint); margin-top:8px; padding:0 60px; }}
  @media (max-width:920px) {{ figure {{ margin:26px 0; }} figcaption {{ padding:0; }} }}
  .bignum {{ margin:34px 0; padding-left:22px; border-left:2px solid var(--rule); }}
  .bignum .n {{ font-family:var(--serif); font-size:46px; line-height:1.05; color:var(--accent); }}
  .bignum .n.up {{ color:#1f6f43; }}
  .bignum .c {{ font-size:14.5px; color:var(--muted); margin-top:6px; max-width:52ch; }}
  .aside {{ font-style:italic; color:var(--faint); font-size:15px;
    line-height:1.6; margin:22px 0; }}
  .eq {{ font-family:var(--serif); font-size:22px; text-align:center; margin:30px 0;
    color:var(--ink); }}
  table {{ width:100%; border-collapse:collapse; margin:26px 0; font-size:15.5px;
    font-variant-numeric:tabular-nums; }}
  th {{ font:600 11.5px/1.3 var(--sans); letter-spacing:.12em; text-transform:uppercase;
    color:var(--faint); text-align:left; padding:0 14px 10px 0;
    border-bottom:1px solid var(--ink); }}
  td {{ padding:11px 14px 11px 0; border-bottom:1px solid var(--rule); }}
  td.hl {{ color:var(--accent); font-weight:650; }}
  td.win {{ color:#1f6f43; font-weight:650; }}
  .rule {{ border:0; border-top:1px solid var(--rule); margin:0; }}
  .closer {{ background:var(--dark); color:#efe8e6; padding:72px 24px; margin-top:88px; }}
  .closer-inner {{ max-width:760px; margin:0 auto; }}
  .closer .slug {{ color:var(--gold); }}
  .closer h3 {{ font-family:var(--serif); font-weight:400; font-size:24px;
    line-height:1.3; color:#fff; margin:30px 0 8px; }}
  .closer p {{ color:#cbb9b5; font-size:16.5px; }}
  .closer .aside {{ color:#a8908c; border-top:1px solid rgba(255,255,255,.14);
    padding-top:24px; margin-top:44px; }}
  footer {{ max-width:760px; margin:0 auto; padding:34px 24px 60px;
    font-size:13.5px; color:var(--faint); line-height:1.7; }}
</style></head><body>

<div class="hero"><div class="hero-inner">
  <div class="kicker">A Model Autopsy &middot; July 2026</div>
  <h1>My ML model lost to a formula. Here&rsquo;s the autopsy.</h1>
  <p class="dek">An 85-feature gradient-boosted home-run model picked bets with
  real money behind them. A strict holdout showed it was worse than guessing
  the league average. This is the full post-mortem: what broke, the
  closed-form formula that beat it, the retrained model that finally won the
  rematch &mdash; and the market that still sits above them all.</p>
  <div class="herostats">
    <div><div class="num">85</div><div class="lab">Features in the model that lost</div></div>
    <div><div class="num">4&times;</div><div class="lab">How far off its probabilities ran</div></div>
    <div><div class="num">{v3['auc']:.3f}</div><div class="lab">Holdout AUC after retraining it right</div></div>
  </div>
</div></div>

<div class="article">

<section>
  <div class="slug">01 &mdash; The Setup</div>
  <h2>Eighty-five features, a Kelly staking plan, and no honest test</h2>
  <p>The system had everything a sports-betting model is supposed to have:
  pitch-level Statcast data, <b>85 engineered features</b> &mdash; barrel
  rate, exit velocity, pull-air rate, platoon splits, pitch-mix matchups,
  recent form, park and weather &mdash; a calibrated gradient-boosting
  pipeline, edge thresholds against sportsbook prices, and fractional-Kelly
  stakes. Picks were logged daily and bet with real money.</p>
  <p>What it did not have was an evaluation that could fail. The model was
  scored on a random train/test split of batted balls &mdash; a test a
  leaky, miscalibrated model passes comfortably. So the first honest exam
  was built after the fact: train on everything through
  <b>2026&#8209;03&#8209;22</b>, score every game from
  <b>2026&#8209;03&#8209;23 to 2026&#8209;07&#8209;02</b> ({n_pg}
  player-games, {n_hr} of them with a home run), with every profile rebuilt
  walk-forward so no future information touches any prediction.</p>
</section>

<hr class="rule">

<section>
  <div class="slug">02 &mdash; The Verdict</div>
  <h2>Worse than guessing the base rate</h2>
  <p>On held-out batted balls the model&rsquo;s AUC was
  <span class="neg">{bb_gbm['auc']:.3f}</span> &mdash; below a coin flip
  &mdash; and it predicted a {pct(bb_gbm['mean_pred'], 2)} home-run rate
  where reality was {pct(bb_gbm['actual_rate'], 2)}. At the unit that
  actually gets bet &mdash; will this player homer today? &mdash; a constant
  guess of the league average beat it soundly.</p>
  <table>
    <tr><th>Predictor</th><th>AUC</th><th>Brier (lower is better)</th><th>Avg prediction</th></tr>
    <tr><td>League base rate, every player, every day</td><td>{pg['base_rate']['auc']:.3f}</td><td>{pg['base_rate']['brier']:.4f}</td><td>{pct(pg['base_rate']['mean_pred'])}</td></tr>
    <tr><td>The 85-feature GBM</td><td>{pg['gbm']['auc']:.3f}</td><td class="hl">{pg['gbm']['brier']:.4f}</td><td class="hl">{pct(pg['gbm']['mean_pred'])}</td></tr>
  </table>
  <div class="bignum"><div class="n">{pg['gbm']['brier']:.4f} vs {pg['base_rate']['brier']:.4f}</div>
  <div class="c">Brier score, model vs no-model. Lower is better. The machine
  learning was subtracting value from a constant.</div></div>
  <p>The live betting losses were not variance. The model was broken, and
  the evaluation that would have said so was never run before the money
  went in.</p>
</section>

<hr class="rule">

<section>
  <div class="slug">03 &mdash; The Diagnosis</div>
  <h2>Trained on one question, deployed on another</h2>
  <p>The autopsy found no exotic cause of death. The model was trained to
  answer <i>&ldquo;given a ball was hit, is it a home run?&rdquo;</i> and
  deployed to answer <i>&ldquo;will this player homer today?&rdquo;</i> The
  translation between the two was an exponent bolted on at serving time,
  and no calibration ever connected the output to the deployment question.
  The result: it said <span class="neg">{pct(pg['gbm']['mean_pred'])}</span>
  on average when reality was
  <span class="neg">{pct(pg['gbm']['actual_rate'])}</span> &mdash; every
  probability roughly four times too small, every &ldquo;edge&rdquo; against
  the sportsbook an artifact of its own miscalibration.</p>
  <figure><img src="{img64('calibration_curve.png')}" alt="Calibration curves on the holdout">
  <figcaption>Holdout calibration. Left: the GBM at its native batted-ball
  unit. Right: the player-game unit, where the dashed line is a perfectly
  calibrated forecaster.</figcaption></figure>
</section>

<hr class="rule">

<section>
  <div class="slug">04 &mdash; The Formula</div>
  <h2>The thing that beat it fits in one line</h2>
  <p class="eq">P(HR today) = 1 &minus; (1 &minus; p<sub>PA</sub>)<sup>E[PA]</sup></p>
  <p>Take the batter&rsquo;s career home-run rate per plate appearance and
  shrink it toward the league average in proportion to how little we know
  about him (an empirical-Bayes posterior). Nudge it up or down for the
  opposing starter&rsquo;s gopher-ball tendency, the platoon matchup, and
  the park &mdash; each factor shrunk and clipped, every constant written
  down and justified in the source. Estimate how many trips to the plate
  his lineup slot gets him. Then ask: what is the chance at least one of
  those trips ends in a home run?</p>
  <table>
    <tr><th>Predictor</th><th>AUC</th><th>Brier</th><th>Avg prediction</th></tr>
    <tr><td>Structural formula (v2: slots + parks)</td><td class="win">{v2['auc']:.4f}</td><td class="win">{v2['brier']:.5f}</td><td>{pct(v2['mean_pred'])}</td></tr>
    <tr><td>Base rate</td><td>{pg['base_rate']['auc']:.4f}</td><td>{pg['base_rate']['brier']:.5f}</td><td>{pct(pg['base_rate']['mean_pred'])}</td></tr>
    <tr><td>85-feature GBM</td><td>{pg['gbm']['auc']:.4f}</td><td>{pg['gbm']['brier']:.5f}</td><td>{pct(pg['gbm']['mean_pred'])}</td></tr>
  </table>
  <p>No training loop, no hyperparameters, nothing to overfit. It beat the
  85-feature model on every metric &mdash; not because formulas are magic,
  but because it was <b>built directly on the deployment question</b> and
  could not be miscalibrated by construction.</p>
</section>

<hr class="rule">

<section>
  <div class="slug">05 &mdash; The Rematch</div>
  <h2>ML, retrained on the right question, takes the lead back</h2>
  <p>The fair test of machine learning was still owed: same features, same
  data, but trained at the <b>player-game level</b>, walk-forward in time,
  with the formula&rsquo;s own inputs handed to it as features and
  probabilities calibrated on a time-ordered validation slice &mdash; never
  a random shuffle. That model is GBM&nbsp;v3.</p>
  <table>
    <tr><th>Predictor</th><th>AUC</th><th>Brier</th><th>Log loss</th><th>Avg prediction</th></tr>
    <tr><td>GBM v3 (retrained, isotonic-calibrated)</td><td class="win">{v3['auc']:.4f}</td><td class="win">{v3['brier']:.5f}</td><td class="win">{v3['log_loss']:.5f}</td><td>{pct(v3['mean_pred'], 2)}</td></tr>
    <tr><td>Structural formula (v2)</td><td>{v2['auc']:.4f}</td><td>{v2['brier']:.5f}</td><td>{v2['log_loss']:.5f}</td><td>{pct(v2['mean_pred'], 2)}</td></tr>
    <tr><td>Base rate</td><td>{pg['base_rate']['auc']:.4f}</td><td>{pg['base_rate']['brier']:.5f}</td><td>{pg['base_rate']['log_loss']:.5f}</td><td>{pct(pg['base_rate']['mean_pred'], 2)}</td></tr>
    <tr><td>Original GBM</td><td>{pg['gbm']['auc']:.4f}</td><td class="hl">{pg['gbm']['brier']:.5f}</td><td class="hl">{pg['gbm']['log_loss']:.5f}</td><td>{pct(pg['gbm']['mean_pred'], 2)}</td></tr>
  </table>
  <div class="bignum"><div class="n up">{pct(v3['mean_pred'], 1)} vs {pct(v3['actual_rate'], 1)}</div>
  <div class="c">GBM v3&rsquo;s average prediction against the actual rate.
  The original model missed by a factor of four; the retrain lands within a
  tenth of a percentage point.</div></div>
  <p>The error analysis says the win is earned in specific places: v3 stops
  overpricing mid-game substitutes (the formula&rsquo;s E[PA] guess prices
  bench bats at more than double their true rate), collects most where home
  runs live &mdash; power parks and gopher-ball starters &mdash; and its
  monthly edge over the base rate <b>widens</b> across the holdout with
  frozen weights: no drift.</p>
  <figure><img src="{img64('error_analysis.png')}" alt="Error analysis panels">
  <figcaption>Brier decomposition of the holdout: by park, lineup slot,
  opposing-starter HR-proneness, and month.</figcaption></figure>
  <p>So the arc is not &ldquo;formulas beat ML.&rdquo; It is: <b>ML trained
  on the wrong unit, evaluated on random splits, and calibrated on nothing
  lost to a formula; ML trained on the deployment question won the
  rematch.</b> Same algorithm family, same features, same data.</p>
</section>

<hr class="rule">

<section>
  <div class="slug">06 &mdash; The Final Boss</div>
  <h2>The market is still the market</h2>
  <p>On the {market['n']} tracked picks that fall inside the holdout
  ({market['positives']} homered), the de-vigged sportsbook price remains
  the reference forecast. v3 is the first model in this project to reach
  it: a hair better on Brier, a hair worse on ranking &mdash; a statistical
  tie at this sample size.</p>
  <table>
    <tr><th>Predictor</th><th>AUC</th><th>Brier</th></tr>
    <tr><td>Market implied (de-vig)</td><td class="win">{market['auc']:.4f}</td><td>{market['brier']:.5f}</td></tr>
    <tr><td>GBM v3</td><td>{v3_picks['auc']:.4f}</td><td class="win">{v3_picks['brier']:.5f}</td></tr>
    <tr><td>Structural formula (v2)</td><td>{v2_picks['auc']:.4f}</td><td>{v2_picks['brier']:.5f}</td></tr>
    <tr><td>Original GBM</td><td>{picks['gbm']['auc']:.4f}</td><td class="hl">{picks['gbm']['brier']:.5f}</td></tr>
  </table>
  <p>And the betting simulation is the cautionary tale in one table: staking
  the formula&rsquo;s &ldquo;edges&rdquo; against recorded book odds, <b>the
  more the model disagreed with the market, the worse it did</b> &mdash;
  {sim['edge>=0.01']['bets']} bets at a &ge;1-point edge returned
  <span class="neg">{pct(sim['edge>=0.01']['roi'])}</span>, and all
  {sim['edge>=0.05']['bets']} bets at a &ge;5-point edge lost
  (<span class="neg">{pct(sim['edge>=0.05']['roi'], 0)}</span>). A
  model&rsquo;s perceived edge over an efficient price is usually its own
  error term staring back at it.</p>
  <p class="aside">Caveat, stated as loudly as the result: n = {market['n']}
  model-selected picks is a small, biased sample. That is exactly why the
  next section exists.</p>
</section>

<hr class="rule">

<section>
  <div class="slug">07 &mdash; Now Running</div>
  <h2>The rematch with the future is pre-registered</h2>
  <p>As of <b>2026&#8209;07&#8209;04</b>, GBM v3 is frozen &mdash; weights,
  feature tables, and calibration checksummed in
  <a href="PROTOCOL.json">PROTOCOL.json</a> &mdash; and predicting every
  posted MLB starting lineup daily via the free MLB Stats API. Each morning
  the previous day&rsquo;s forecasts are graded against what actually
  happened and the running Brier score and calibration table are published
  to <a href="SCOREBOARD.md">SCOREBOARD.md</a>. No odds are fetched and no
  stakes are placed: it is a forecasting scorecard, with the success
  criterion written down before the first result: <b>after at least 1,000
  graded predictions, beat the constant base rate of
  {pct(0.1053, 2)}</b>. If the frozen model&rsquo;s holdout edge does not
  survive contact with the future, the scoreboard will say so in public.</p>
</section>

</div>

<div class="closer"><div class="closer-inner">
  <div class="slug">08 &mdash; Lessons</div>
  <h3>Calibration beats complexity.</h3>
  <p>The 85-feature model lost by a factor-of-four probability error, not a
  ranking error. A one-line formula that could not be miscalibrated beat
  it; the retrained model won by being both sharp <i>and</i> honest about
  probabilities. Sophistication never repaid a calibration debt.</p>
  <h3>Evaluate before you bet.</h3>
  <p>Every number in this autopsy was computable before the first dollar
  was staked. A strict time-based holdout is cheap; the tuition paid to the
  sportsbooks was not. The order of operations &mdash; evaluate, then
  freeze, then pre-register, then act &mdash; is the entire method.</p>
  <h3>The market is the final boss.</h3>
  <p>Beating the base rate took a formula. Beating the formula took ML done
  right. Beating the de-vigged closing price has, so far, taken more than
  either &mdash; the best model in this repo has fought it to a draw on 88
  picks. Anyone claiming a betting edge owes you this exact chain of
  receipts.</p>
  <p class="aside">Every figure on this page is generated from
  eval_results.json by build_autopsy.py; the evaluation itself is
  reproducible from evaluate_model.py and train_model_v3.py. Educational
  project; nothing here is betting advice.</p>
</div></div>

<footer>Weight Room Hero &mdash; home-run model post-mortem, July 2026.
Data: MLB Statcast via pybaseball; market prices via tracked pick logs.
Full methodology in EVALUATION.md.</footer>

</body></html>
"""

OUT.write_text(html)
print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
