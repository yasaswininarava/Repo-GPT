# Start Chat with GitHub and open it in Google Chrome.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$chrome = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $chrome) {
    Write-Error "Google Chrome was not found. Install Chrome, then run this script again."
    exit 1
}

# Streamlit uses this variable when it opens a browser on Windows.
$env:BROWSER = $chrome
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $python) {
    & $python -m streamlit run app.py
} else {
    streamlit run app.py
}
