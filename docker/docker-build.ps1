<#
.SYNOPSIS
  Build the sam2-trainer:dev image (build-phase counterpart to run-cursor-dev.ps1).

.DESCRIPTION
  Builds docker/Dockerfile from the Sam2SegmentationTrainer repository root and tags
  sam2-trainer:dev. Separating this from run-cursor-dev.ps1 keeps image build-args
  out of the Compose/CIFS run path.

  Merge optional KEY=VALUE files from the invocation directory (not the repo):
    build.env
    .build.sam2-trainer-dev.env
  Later files override earlier keys. Then --build-arg on the command line from those keys.
  Committed docker/example.sam2-trainer-dev.build.env is a template only and is not read.

  Invoke from D:\Docker\Builds\sam2-trainer so machine-local build.env files stay out of git.

  After a successful tag, recreate a running Compose cursor-dev (Dev Container)
  so it uses the new image ID. docker build alone leaves the old container in place.
  One-off compose run sessions are not replaced; the next run-cursor-dev.ps1 uses the new image.

.PARAMETER NoCache
  Pass --no-cache to docker build.

.PARAMETER SkipRecreate
  Build and tag only; do not replace a running cursor-dev container.

.PARAMETER RecreateOnly
  Skip docker build; recreate a running cursor-dev onto the current sam2-trainer:dev tag.

.PARAMETER ExtraArgs
  Additional docker build arguments appended after the merged --build-arg list.
#>
param(
    [switch]$NoCache,

    [switch]$SkipRecreate,

    [switch]$RecreateOnly,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

if ($RecreateOnly -and $SkipRecreate) {
    Write-Error 'Use either -RecreateOnly or -SkipRecreate, not both.'
}

function Read-DotEnvFile {
    param([Parameter(Mandatory)][string]$Path)
    $map = [ordered]@{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $map
    }
    Write-Host "Found and merged: $Path"
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) {
            return
        }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) {
            return
        }
        $key = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        $map[$key] = $value
    }
    return $map
}

function Get-Sam2RunningCursorDevIds {
    param([Parameter(Mandatory)][string]$DockerExe)
    $found = [System.Collections.Generic.List[string]]::new()
    $labeled = & $DockerExe ps -q `
        --filter 'label=com.docker.compose.project=sam2-trainer' `
        --filter 'label=com.docker.compose.service=cursor-dev'
    foreach ($id in @($labeled)) {
        $trim = [string]$id
        if (-not [string]::IsNullOrWhiteSpace($trim)) {
            [void]$found.Add($trim.Trim())
        }
    }
    # After retag, ancestor=sam2-trainer:dev matches the new ID only. Image name
    # on the old container is still sam2-trainer:dev.
    $rows = & $DockerExe ps --format '{{.ID}}|{{.Image}}|{{.Names}}'
    foreach ($row in @($rows)) {
        if ([string]::IsNullOrWhiteSpace($row)) { continue }
        $parts = $row.Split('|')
        if ($parts.Count -lt 3) { continue }
        $id = $parts[0]
        $image = $parts[1]
        $names = $parts[2]
        if ($image -ne 'sam2-trainer:dev' -and $names -notlike '*sam2-trainer*cursor-dev*') {
            continue
        }
        if (-not $found.Contains($id)) {
            [void]$found.Add($id)
        }
    }
    return , $found.ToArray()
}

function Invoke-Sam2CursorDevRecreate {
    param(
        [Parameter(Mandatory)][string]$DockerExe,
        [Parameter(Mandatory)][string]$ImageTag,
        [Parameter(Mandatory)][string]$DockerDir,
        [Parameter(Mandatory)][string]$RepoRoot
    )
    $ids = Get-Sam2RunningCursorDevIds -DockerExe $DockerExe
    if ($ids.Count -eq 0) {
        Write-Host "No running cursor-dev to replace. Start or Reopen in Container to use $ImageTag."
        return 0
    }
    $recreated = $false
    $seen = @{}
    foreach ($id in $ids) {
        $raw = & $DockerExe inspect $id --format '{{json .Config.Labels}}'
        $labels = $raw | ConvertFrom-Json
        $oneoff = [string]$labels.'com.docker.compose.oneoff'
        if ($oneoff -eq 'True' -or $oneoff -eq 'true') {
            Write-Host "Skipping one-off container $id; next run-cursor-dev.ps1 will use $ImageTag."
            continue
        }
        $project = [string]$labels.'com.docker.compose.project'
        $service = [string]$labels.'com.docker.compose.service'
        $files = [string]$labels.'com.docker.compose.project.config_files'
        $workdir = [string]$labels.'com.docker.compose.project.working_dir'
        if ([string]::IsNullOrWhiteSpace($project)) { $project = 'sam2-trainer' }
        if ([string]::IsNullOrWhiteSpace($service)) { $service = 'cursor-dev' }
        $key = "$project|$service|$files|$workdir"
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true

        $composeArgs = @('compose', '-p', $project)
        if (-not [string]::IsNullOrWhiteSpace($workdir)) {
            $composeArgs += @('--project-directory', $workdir)
        }
        else {
            $composeArgs += @('--project-directory', $RepoRoot)
        }
        if (-not [string]::IsNullOrWhiteSpace($files)) {
            foreach ($f in ($files -split ',')) {
                $trim = $f.Trim()
                if ($trim) { $composeArgs += @('-f', $trim) }
            }
        }
        else {
            $composeArgs += @('-f', (Join-Path $DockerDir 'compose.cursor-dev.yaml'))
            $envFile = Join-Path $DockerDir '.env'
            if (Test-Path -LiteralPath $envFile) {
                $composeArgs += @('--env-file', $envFile)
            }
            $userRoot = $env:NORNIR_DOCKER_USER_ROOT
            if ([string]::IsNullOrWhiteSpace($userRoot)) { $userRoot = 'D:\Docker' }
            foreach ($rel in @(
                    'Run\sam2-dev\compose.net-mounts.override.yaml',
                    'Run\nornir-dev\compose.net-mounts.override.yaml'
                )) {
                $override = Join-Path $userRoot $rel
                if (Test-Path -LiteralPath $override) {
                    $composeArgs += @('-f', $override)
                    break
                }
            }
        }
        $composeArgs += @('up', '-d', '--force-recreate', '--no-build', '--no-deps', $service)
        Write-Host "Recreating $service (project $project) onto $ImageTag"
        Write-Host "$DockerExe $($composeArgs -join ' ')"
        & $DockerExe @composeArgs
        if ($LASTEXITCODE -ne 0) { return [int]$LASTEXITCODE }
        $recreated = $true
    }
    if ($recreated) {
        Write-Host "Replaced running cursor-dev with $ImageTag. If Cursor is attached, wait for reconnect or Reopen in Container."
    }
    return 0
}

$ImageTag = 'sam2-trainer:dev'
$ImageId = $ImageTag -replace ':', '-'
$BuildEnvRoot = (Get-Location).ProviderPath
$DockerDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $DockerDir

Get-ChildItem -LiteralPath $BuildEnvRoot -Filter 'example.*.build.env' -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host ("Template name not merged: {0}`n  Copy to {1}" -f $_.FullName, (Join-Path $BuildEnvRoot ".build.$ImageId.env"))
    }

$merged = [ordered]@{}
$mergeNames = @('build.env', ".build.$ImageId.env")
# Older layout used the image name without the :dev suffix.
if (Test-Path -LiteralPath (Join-Path $BuildEnvRoot '.build.sam2-trainer.env')) {
    $mergeNames += '.build.sam2-trainer.env'
}
foreach ($name in $mergeNames) {
    $path = Join-Path $BuildEnvRoot $name
    if (Test-Path -LiteralPath $path) {
        $part = Read-DotEnvFile -Path $path
        foreach ($k in $part.Keys) {
            $merged[$k] = $part[$k]
        }
    }
    else {
        Write-Host "Not found (skipped): $path"
    }
}

# Relative -f and context after cd. An absolute Windows path (D:\...) is parsed by
# buildx as a URL scheme "D:", which drops the context. Skip empty ExtraArgs: a
# null splat from ./build.ps1 becomes "" and that empty PATH triggers
# "'docker buildx build' requires 1 argument".
$buildArgs = @(
    'build',
    '-f', 'docker/Dockerfile',
    '-t', $ImageTag
)
if ($NoCache) {
    $buildArgs += '--no-cache'
}
foreach ($k in $merged.Keys) {
    $buildArgs += @('--build-arg', "$k=$($merged[$k])")
}
if ($null -ne $ExtraArgs) {
    foreach ($a in $ExtraArgs) {
        if ([string]::IsNullOrWhiteSpace($a)) {
            continue
        }
        $buildArgs += $a
    }
}
$buildArgs += @('--', '.')

$dockerCmd = Get-Command docker.exe -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $dockerCmd) {
    $dockerCmd = Get-Command docker -CommandType Application -ErrorAction Stop |
        Select-Object -First 1
}

if (-not $RecreateOnly) {
    Write-Host "Building ${ImageTag} (context $RepoRoot, env from $BuildEnvRoot)"
    Write-Host "$($dockerCmd.Source) $($buildArgs -join ' ')"
    $buildExit = 1
    Push-Location -LiteralPath $RepoRoot
    try {
        & $dockerCmd.Source @buildArgs
        $buildExit = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($buildExit -ne 0) {
        exit $buildExit
    }
}
if ($SkipRecreate) {
    Write-Host "SkipRecreate: $ImageTag tagged; a running cursor-dev still uses the previous image ID."
    exit 0
}
exit (Invoke-Sam2CursorDevRecreate -DockerExe $dockerCmd.Source -ImageTag $ImageTag -DockerDir $DockerDir -RepoRoot $RepoRoot)
