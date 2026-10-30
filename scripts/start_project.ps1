$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw "Python virtual environment not found at $venvPython. Create it first with: py -3.11 -m venv .venv"
}

$backendStdout = Join-Path $projectRoot 'backend-start.stdout.log'
$backendStderr = Join-Path $projectRoot 'backend-start.stderr.log'
$frontendStdout = Join-Path $projectRoot 'frontend-start.stdout.log'
$frontendStderr = Join-Path $projectRoot 'frontend-start.stderr.log'

$backendProcess = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python.*' -and $_.CommandLine -match 'uvicorn.*backend\.app\.main:app'
} | Select-Object -First 1

if (-not $backendProcess) {
    $backendProcess = Start-Process -FilePath $venvPython -ArgumentList '-m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000' -WorkingDirectory $projectRoot -RedirectStandardOutput $backendStdout -RedirectStandardError $backendStderr -PassThru
}

$ready = $false
for ($attempt = 1; $attempt -le 40; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    throw 'FastAPI did not become ready on http://127.0.0.1:8000 within the timeout.'
}

$frontendCommand = 'npm run dev -- --host 127.0.0.1 --port 5173'
$frontendProcess = Start-Process -FilePath 'cmd.exe' -ArgumentList "/c cd /d `"$projectRoot\frontend`" && $frontendCommand" -RedirectStandardOutput $frontendStdout -RedirectStandardError $frontendStderr -PassThru

Write-Host 'Backend is running at http://127.0.0.1:8000'
Write-Host 'Frontend is starting at http://127.0.0.1:5173'
Write-Host "Backend PID: $($backendProcess.Id)"
Write-Host "Frontend PID: $($frontendProcess.Id)"
Write-Host 'Use Ctrl+C in the launching terminal to stop both processes if needed.'
