<#
.SYNOPSIS
  Build the sam2-trainer:dev image (build-phase counterpart to run-cursor-dev.ps1).

.DESCRIPTION
  Builds docker/Dockerfile from the Sam2Trainer repository root and tags
  sam2-trainer:dev. Separating this from run-cursor-dev.ps1 keeps image build-args
  out of the Compose/CIFS run path.

  Merge optional KEY=VALUE files from the invocation directory (not the repo):
    build.env
    .build.sam2-trainer-dev.env
  Later files override earlier keys. Then --build-arg on the command line from those keys.
  Committed docker/example.sam2-trainer-dev.build.env is a template only and is not read.

  Invoke from D:\Docker\Builds\sam2-trainer so machine-local build.env files stay out of git.

  After a successful tag, bring the Compose cursor-dev to the chosen modality (and start both galleries and gallery-router)
  on the new image: started when not running, recreated when needed. docker build alone leaves the
  old container in place. One-off compose run sessions are not replaced; the next
  run-cursor-dev.ps1 uses the new image.

.PARAMETER Modality
  TEM or SEM (first argument: build.ps1 SEM). Required unless -SkipRecreate. The image is the same
  for both; the modality picks which env file is layered over docker/.env for the containers that
  are started or recreated (<Modality>.env: data mirror, output folder, checkpoint root).
  After the build, cursor-dev and the annotation gallery are brought to that modality on the new
  image: started if not running, recreated if the image or the modality differs, left alone if
  already right. If a cursor-dev of the other modality is running it asks
  whether to replace it (Yes recreates it as the requested modality, No leaves it and exits).

.PARAMETER NoCache
  Pass --no-cache to docker build.

.PARAMETER SkipRecreate
  Build and tag only; do not replace a running cursor-dev container.

.PARAMETER ExtraArgs
  Additional docker build arguments appended after the merged --build-arg list.
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('TEM', 'SEM')]
    [string]$Modality,

    [switch]$NoCache,

    [switch]$SkipRecreate,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

if (-not $SkipRecreate -and [string]::IsNullOrWhiteSpace($Modality)) {
    [Console]::Error.WriteLine('error: pass the modality as the first argument: rebuild-and-start.ps1 TEM or SEM (or -SkipRecreate to only build).')
    exit 1
}

. (Join-Path $PSScriptRoot 'sam2-modality.ps1')

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
        [Parameter(Mandatory)][string]$RepoRoot,
        [Parameter(Mandatory)][string]$Modality,
        [Parameter(Mandatory)][string]$ModalityEnv
    )
    $modalityEnv = $ModalityEnv
    $sharedEnv = Join-Path $DockerDir '.env'
    $userRoot = $env:NORNIR_DOCKER_USER_ROOT
    if ([string]::IsNullOrWhiteSpace($userRoot)) { $userRoot = 'D:\Docker' }
    $newImageId = ([string](& $DockerExe image inspect $ImageTag --format '{{.Id}}' 2>$null)).Trim()

    # Compose arguments from the repo files, as Reopen in Container would use them.
    $defaultArgs = @('compose', '-p', 'sam2-trainer', '--project-directory', $DockerDir,
        '-f', (Join-Path $DockerDir 'compose.cursor-dev.yaml'))
    foreach ($rel in @('Run\sam2-dev\compose.net-mounts.override.yaml', 'Run\nornir-dev\compose.net-mounts.override.yaml')) {
        $override = Join-Path $userRoot $rel
        if (Test-Path -LiteralPath $override) { $defaultArgs += @('-f', $override); break }
    }
    if (Test-Path -LiteralPath $sharedEnv) { $defaultArgs += @('--env-file', $sharedEnv) }
    $defaultArgs += @('--env-file', $modalityEnv)

    # ---- cursor-dev ----
    $ids = Get-Sam2RunningCursorDevIds -DockerExe $DockerExe
    $handled = $false
    $cursorChanged = $false
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
        $handled = $true

        # Needed when the image changed or the modality differs. A container without
        # SAM2_MODALITY predates this scheme and counts as TEM.
        $containerImage = ([string](& $DockerExe inspect $id --format '{{.Image}}' 2>$null)).Trim()
        $containerModality = 'TEM'
        foreach ($l in @(& $DockerExe inspect $id --format '{{range .Config.Env}}{{println .}}{{end}}' 2>$null)) {
            if ($l -like 'SAM2_MODALITY=*') { $containerModality = $l.Substring('SAM2_MODALITY='.Length).Trim() }
        }
        $reasons = @()
        if ($containerModality -ne $Modality) { $reasons += "modality $containerModality -> $Modality" }
        if ($containerImage -ne $newImageId) { $reasons += 'new image' }
        if ($reasons.Count -eq 0) {
            Write-Host "$service is already $Modality on the current image; leaving it running."
            continue
        }

        $composeArgs = @('compose', '-p', $project)
        if (-not [string]::IsNullOrWhiteSpace($workdir)) {
            $composeArgs += @('--project-directory', $workdir)
        }
        else {
            $composeArgs += @('--project-directory', $DockerDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($files)) {
            foreach ($f in ($files -split ',')) {
                $trim = $f.Trim()
                if ($trim) { $composeArgs += @('-f', $trim) }
            }
        }
        else {
            $composeArgs = $defaultArgs[0..($defaultArgs.IndexOf('--env-file') - 1)]
        }
        # Always pass the shared env and the modality env: compose does not load them for a
        # project when -f comes from the container labels.
        if (Test-Path -LiteralPath $sharedEnv) {
            $composeArgs += @('--env-file', $sharedEnv)
        }
        $composeArgs += @('--env-file', $modalityEnv)
        $composeArgs += @('up', '-d', '--force-recreate', '--no-build', '--no-deps', $service)
        Write-Host "Recreating $service (project $project, $Modality; $($reasons -join ', ')) onto $ImageTag"
        Write-Host "$DockerExe $($composeArgs -join ' ')"
        & $DockerExe @composeArgs
        if ($LASTEXITCODE -ne 0) { return [int]$LASTEXITCODE }
        $cursorChanged = $true
    }
    if (-not $handled) {
        $upArgs = $defaultArgs + @('up', '-d', '--no-build', '--no-deps', 'cursor-dev')
        Write-Host "No running cursor-dev; starting it ($Modality) on $ImageTag"
        Write-Host "$DockerExe $($upArgs -join ' ')"
        & $DockerExe @upArgs
        if ($LASTEXITCODE -ne 0) { return [int]$LASTEXITCODE }
        $cursorChanged = $true
    }
    if ($cursorChanged) {
        Write-Host "cursor-dev is up as $Modality on $ImageTag. If Cursor is attached, wait for reconnect or Reopen in Container."
    }

    # ---- galleries: TEM and SEM run side by side; only the gallery.codepharm.net redirect follows the modality ----
    $galleryArgs = $defaultArgs + @('up', '-d', '--no-build', '--no-deps', 'annotation-gallery-tem', 'annotation-gallery-sem', 'gallery-router')
    Write-Host "Starting the TEM and SEM galleries and gallery-router (gallery.codepharm.net -> $Modality)"
    Write-Host "$DockerExe $($galleryArgs -join ' ')"
    & $DockerExe @galleryArgs
    if ($LASTEXITCODE -ne 0) { return [int]$LASTEXITCODE }
    Set-Sam2ActiveGalleryModality -DockerExe $DockerExe -Modality $Modality -EnvFile $sharedEnv
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

# Pre-flight before the (long) build: the modality's data mirror must exist, and the other modality's
# trainer, if running, must be confirmed for replacement. A missing mirror or a "No" costs no build.
$ModalityEnv = $null
if (-not $SkipRecreate) {
    try {
        Confirm-Sam2ReplaceOtherModality -DockerExe $dockerCmd.Source -Modality $Modality
        $ModalityEnv = Initialize-Sam2Modality -Modality $Modality -DockerDir $DockerDir
    }
    catch {
        # One line, no PowerShell stack trace: these are instructions, not crashes.
        [Console]::Error.WriteLine("error: $($_.Exception.Message)")
        exit 1
    }
}

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
if ($SkipRecreate) {
    Write-Host "SkipRecreate: $ImageTag tagged; a running cursor-dev still uses the previous image ID."
    exit 0
}
exit (Invoke-Sam2CursorDevRecreate -DockerExe $dockerCmd.Source -ImageTag $ImageTag -DockerDir $DockerDir -RepoRoot $RepoRoot -Modality $Modality -ModalityEnv $ModalityEnv)
