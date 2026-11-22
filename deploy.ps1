param([Parameter(Mandatory, Position=0)][string]$Target, [switch]$StartServer)
$ErrorActionPreference = "Stop"
if ($Target -match '^USER@|@PI_IP$') {
    throw "Use the actual Pi login and IP, for example the same login you use at the Pi console."
}
$dirs = @(
    (Get-Command ssh.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
    "$env:WINDIR\Sysnative\OpenSSH\ssh.exe",
    "$env:WINDIR\System32\OpenSSH\ssh.exe",
    "$env:ProgramFiles\Git\usr\bin\ssh.exe"
)
foreach ($path in $dirs) {
    if ($path -and (Test-Path -LiteralPath $path)) {
        $scp = Join-Path (Split-Path $path) 'scp.exe'
        if (Test-Path -LiteralPath $scp) { $ssh = $path; break }
    }
}
if (-not $ssh) { throw "Install Windows OpenSSH Client or Git for Windows, then rerun this script." }
Push-Location $PSScriptRoot
try {
    & $ssh $Target 'mkdir -p asl-demo/pi asl-demo/src asl-demo/models'
    if ($LASTEXITCODE) { throw "Could not create Pi directories" }
    foreach ($group in @('pi', 'src', 'models')) {
        $files = switch ($group) {
            pi { 'pi/server.py' }
            src { 'src/frame_protocol.py', 'src/mlp_numpy.py', 'src/normalization.py' }
            models { 'models/mlp.npz' }
        }
        & $scp @files "${Target}:asl-demo/$group/"
        if ($LASTEXITCODE) { throw "Copy to $group failed" }
    }
    if ($StartServer) {
        & $ssh -t $Target 'cd asl-demo && python3 pi/server.py'
        if ($LASTEXITCODE) { throw "Pi server SSH session exited with code $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}

