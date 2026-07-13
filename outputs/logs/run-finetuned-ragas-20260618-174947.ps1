$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\nicoh\OneDrive\Dev\Proyecto Investigacion\PEFT\medical-peft'
function Log([string]$msg) {
  $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
  Add-Content -LiteralPath 'C:\Users\nicoh\OneDrive\Dev\Proyecto Investigacion\PEFT\medical-peft\outputs\logs\finetuned-ragas-20260618-174947.log' -Value $line
  Write-Output $line
}
function Run-Step([string]$name, [string[]]$argsList) {
  Log "START: $name"
  Log "COMMAND: C:\Users\nicoh\OneDrive\Dev\Proyecto Investigacion\PEFT\medical-peft\.venv312\Scripts\python.exe $($argsList -join ' ')"
  & 'C:\Users\nicoh\OneDrive\Dev\Proyecto Investigacion\PEFT\medical-peft\.venv312\Scripts\python.exe' @argsList *>&1 | ForEach-Object { Add-Content -LiteralPath 'C:\Users\nicoh\OneDrive\Dev\Proyecto Investigacion\PEFT\medical-peft\outputs\logs\finetuned-ragas-20260618-174947.log' -Value $_.ToString(); Write-Output $_.ToString() }
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $name exit=$LASTEXITCODE" }
  Log "DONE: $name"
}
Log 'Fine-tuned RAGAS evaluation started'
Run-Step 'Evaluate Gemma4 grounded / QLoRA' @('scripts/evaluate_model_predictions.py','--input','outputs/gemma4-grounded/test_predictions.jsonl','--output','outputs/gemma4-grounded/test_eval.json','--max-completion-tokens','4096','--resume')
Run-Step 'Evaluate MedGemma grounded / QLoRA' @('scripts/evaluate_model_predictions.py','--input','outputs/medgemma-grounded/test_predictions.jsonl','--output','outputs/medgemma-grounded/test_eval.json','--max-completion-tokens','4096','--resume')
Log 'ALL DONE'
