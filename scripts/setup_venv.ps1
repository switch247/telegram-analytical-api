<#
Create a venv at ./venv and install project dependencies.

Run from the repository root in PowerShell:

    .\scripts\setup_venv.ps1

This will:
 - create a venv in `./venv` (Python 3.8+ required)
 - activate it for the current session
 - upgrade pip and install dependencies from `requirements.txt`
 - install `psycopg2-binary` to ease Windows installs
#>

param(
    [string]$VenvPath = './.venv',
    [string]$Requirements = './requirements.txt'
)

Write-Host "Creating virtual environment at $VenvPath ..."
python -m venv $VenvPath

Write-Host "Activating virtual environment..."
$venvFullPath = Resolve-Path $VenvPath
. (Join-Path $venvFullPath "Scripts/Activate.ps1")

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

if (Test-Path $Requirements) {
    Write-Host "Installing dependencies from $Requirements..."
    python -m pip install -r $Requirements
} else {
    Write-Host "Requirements file not found at $Requirements; skipping."
}

Write-Host "Installing psycopg2-binary for Windows compatibility..."
python -m pip install psycopg2-binary

Write-Host "Setup complete. To activate the venv in a new session run:`n .\venv\Scripts\Activate.ps1"
