# Ubuntu API and MCP verification

Verification date: 19 October 2026. Tested source commit: `3fbca3b672ed598333db33d2630b72bffe4ff3b0`.

## Outcomes

- All 20 automated API/MCP tests passed in 1.224 seconds.
- REST completed real Pathum Thani weather and water-level forecasts.
- The official Python MCP client exercised all seven tools and completed both forecasts.
- Selected Ridge validation and test metrics matched across REST and MCP. The water comparison is recorded in `api-comparison.json`.
- Codex CLI retrieved an existing experiment and submitted one new water forecast through MCP. Its saved run completed with no reported error.

| Task | Horizon | Selected model | Test MAE | Test RMSE | Test MASE |
| --- | --- | --- | --- | --- | --- |
| Weather maximum temperature | 1 day | Ridge | 1.020743 C | 1.334933 C | 0.428646 |
| Water level station S1 | 24 hours | Ridge | 0.095486 | 0.134186 | 0.717235 |

Water error units have not been independently confirmed. MASE uses a training-period seasonal-change reference, not a separately evaluated test baseline.

## Evidence layout

- `verification/report-evidence/`: automated test log, runtime details, package inventory and Conda export.
- `verification/weather-api-*` and `verification/water-api-*`: REST results, metrics and reports.
- `verification/weather-mcp-*` and `verification/water-mcp-*`: official-client results and reports; water also includes request, profile, status and metric comparison.
- `verification/codex-water-run/`: saved configuration, completed run record, data profile, result and report from the Codex-submitted run.

The package inventory and Conda export are provenance snapshots, not portable lockfiles. They contain machine-specific installation paths. Use the project's installation documentation to create a fresh environment.

## Scope and limitations

Clients and forecasting execution ran on the same Ubuntu host. Local execution disabled CrewAI agent review. Codex supplied tool-calling decisions using a user-specified configuration. Ollama Cloud, live LANTA submission, autonomous model design, remote multi-user access and production service operation were not tested here.

Weather REST compared Ridge and random forest plus baselines; weather MCP requested Ridge plus baselines. Matching Ridge metrics do not mean identical model searches.

The water upload contained 284,581 observations and no missing cell values. Timeline regularization revealed 18,084 missing timestamps. Inputs were forward-filled from earlier observations and missing future labels were excluded. Gap sensitivity still needs further evaluation.

Raw observations, trained models, predictions, credentials and the duplicate compressed evidence archive are not included in this evidence commit. Screenshots and the detailed supervisor report are maintained separately.
