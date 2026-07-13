param(
    [switch]$SkipBaseInference,
    [switch]$SkipEvaluation,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "outputs\logs"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$MainLog = Join-Path $LogDir "overnight-model-evaluation-$Timestamp.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $MainLog -Value $line -Encoding UTF8
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Log "START: $Name"
    Write-Log "COMMAND: $Python $($Arguments -join ' ')"

    if ($DryRun) {
        Write-Log "DRY-RUN: skipped execution"
        return
    }

    Push-Location $ProjectRoot
    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $Python
        $escapedArgs = $Arguments | ForEach-Object {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        $psi.Arguments = $escapedArgs -join " "
        $psi.WorkingDirectory = $ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        $process.EnableRaisingEvents = $true

        $outputHandler = [System.Diagnostics.DataReceivedEventHandler]{
            param($sender, $eventArgs)
            if ($null -ne $eventArgs.Data) {
                Write-Host $eventArgs.Data
                Add-Content -Path $MainLog -Value $eventArgs.Data -Encoding UTF8
            }
        }
        $errorHandler = [System.Diagnostics.DataReceivedEventHandler]{
            param($sender, $eventArgs)
            if ($null -ne $eventArgs.Data) {
                Write-Host $eventArgs.Data
                Add-Content -Path $MainLog -Value $eventArgs.Data -Encoding UTF8
            }
        }
        $process.add_OutputDataReceived($outputHandler)
        $process.add_ErrorDataReceived($errorHandler)

        [void]$process.Start()
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        $process.WaitForExit()
        $process.WaitForExit()

        if ($process.ExitCode -ne 0) {
            throw "Step failed with exit code $($process.ExitCode)"
        }
        $process.remove_OutputDataReceived($outputHandler)
        $process.remove_ErrorDataReceived($errorHandler)
    }
    finally {
        Pop-Location
    }

    Write-Log "DONE: $Name"
}

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}

Write-Log "Project root: $ProjectRoot"
Write-Log "Python: $Python"
Write-Log "Log file: $MainLog"

Invoke-Step "Verify Python environment" @(
    "-c",
    "import sys, torch; print(sys.executable); print('torch_cuda', torch.cuda.is_available())"
)

Invoke-Step "Verify evaluation dependencies" @(
    "-c",
    "import ragas, openai, langchain_community; print('ragas_ok', getattr(ragas, '__version__', 'no-version'), langchain_community.__version__)"
)

if (-not $SkipBaseInference) {
    Invoke-Step "Generate MedGemma base predictions" @(
        "scripts/inference_base.py",
        "--model-name", "google/medgemma-1.5-4b-it",
        "--output-prefix", "outputs/medgemma-base/test"
    )
}

if (-not $SkipEvaluation) {
    $evaluations = @(
        @{
            Name = "Evaluate Gemma4 base"
            Input = "outputs/gemma4-base/test_predictions.jsonl"
            Output = "outputs/gemma4-base/test_eval.json"
        },
        @{
            Name = "Evaluate Gemma4 QLoRA"
            Input = "outputs/gemma4-grounded/test_predictions.jsonl"
            Output = "outputs/gemma4-grounded/test_eval.json"
        },
        @{
            Name = "Evaluate MedGemma base"
            Input = "outputs/medgemma-base/test_predictions.jsonl"
            Output = "outputs/medgemma-base/test_eval.json"
        },
        @{
            Name = "Evaluate MedGemma QLoRA"
            Input = "outputs/medgemma-grounded/test_predictions.jsonl"
            Output = "outputs/medgemma-grounded/test_eval.json"
        }
    )

    foreach ($eval in $evaluations) {
        $inputPath = Join-Path $ProjectRoot $eval.Input
        if (-not (Test-Path $inputPath)) {
            Write-Log "SKIP: $($eval.Name) because input does not exist: $($eval.Input)"
            continue
        }

        Invoke-Step $eval.Name @(
            "scripts/evaluate_model_predictions.py",
            "--input", $eval.Input,
            "--output", $eval.Output,
            "--max-completion-tokens", "2048"
        )
    }
}

Write-Log "ALL DONE"
