# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""End-to-end harness.run on REAL model artifacts (the integration test the verifiers flagged)."""
import json, sys, glob, os
from calib_harness import run

# Run tree with ModelTeam/<mode>/run_001/{model_design,calibration}_output.json
BASE=os.path.join(os.environ.get("AEL_CALIB_RUNS_ROOT", "<runs_root>"), "ModelTeam")

def check(run_dir):
    design=json.load(open(f"{run_dir}/model_design_output.json"))['formal_models']
    calib=json.load(open(f"{run_dir}/calibration_output.json"))['calibrated_models']
    tag=run_dir.split('/ModelTeam/')[1]
    for i,fm in enumerate(design):
        cps = calib[i].get('calibrated_parameters',[]) if i<len(calib) else []
        oc = run(fm, cps)
        fs = f"{oc.fit_score:.3f}" if oc.fit_score is not None else "None"; mm = f"{oc.moment_match_score:.3f}" if oc.moment_match_score is not None else "None"
        cov=oc.coverage
        print(f"  [{tag} M{i}] {oc.calibration_status:20} fit={fs:6} match={mm:6} df={oc.degrees_of_freedom} "
              f"| eqs {cov.get('parseable_eqs')}ok/{cov.get('refused_eqs')}ref "
              f"over_id={cov.get('over_id_moments')} pin={cov.get('pin_moments')} taut={cov.get('tautological_moments')}")
        if oc.uncalibratable_reason: print(f"        reason: {oc.uncalibratable_reason}")
        for r in oc.moments:
            if r.role=='over_id':
                print(f"        over_id {r.moment_key}: computed={r.computed:.4f} target={r.target} rel_err={r.rel_error:.2%} [{r.source_citation[:30]}]")

n=0
for run_dir in sorted(glob.glob(f"{BASE}/*/run_001")):
    print(f"=== {run_dir.split('/ModelTeam/')[1]} ===")
    try: check(run_dir); n+=1
    except Exception as e:
        import traceback; print("  ERROR:", type(e).__name__, str(e)[:120]); traceback.print_exc()
print(f"\nran {n} run-dirs, no crash" if n else "no data")
