# Copy videos off the iPhone over MTP, and actually verify each one arrived.
#
# The trap this exists for: a destination file appears at its FULL SIZE almost
# immediately while the copy is still streaming, so "the file exists and is
# 4.5 GB" proves nothing. A previous session reported a copy as finished on
# exactly that basis and was wrong. The only reliable signal is the size holding
# steady across several checks with the file no longer locked for writing.
#
# 2026-08-01: size stability is ALSO not enough. A stalled copy -- the phone
# locking mid-transfer will do it -- leaves the file at full pre-allocated size,
# not growing, and completely unreadable. It looks finished by every size-based
# test. An iPhone MOV keeps its index (the moov atom) at the END of the file, so
# a truncated copy has the bytes but no index, and the only honest check is
# whether the video actually opens and reports a frame count. That check is now
# run by verify-footage.py after this script, and nothing downstream starts
# until it passes.
#
# Keep the phone UNLOCKED for the whole transfer. Auto-Lock ends it silently.

# Hardcoded rather than parameterised. A param block didn't bind when the script
# was launched with -File or dot-sourced from -Command. The REAL cause of the
# repeated "ALBUM NOT FOUND", though, was mine and had nothing to do with params:
# PowerShell variable names are case-INSENSITIVE, so `$album = $null` silently
# wiped `$Album`, the name being searched for, and every folder was then compared
# against null. Renamed so the two can't collide.
#
# Which album and which files, though, change every session, and editing the
# script each time is how it ended up chasing a name that was two transfers old.
# Environment variables DO bind under -File, unlike a param block, so they are
# the way to pass them: run list-phone.ps1 first, then
#   PULL_ALBUM=202608_a PULL_NAMES=IMG_2932.MOV powershell -File scripts/pull-footage.ps1
# The literals below stay only as the fallback.
$AlbumName = if ($env:PULL_ALBUM) { $env:PULL_ALBUM } else { "202608_a" }
$Dest  = "C:\dev\poolvision\footage"
$Names = if ($env:PULL_NAMES) { $env:PULL_NAMES -split "," } else { @("IMG_2770.MOV") }

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$sh = New-Object -ComObject Shell.Application
$dev = $sh.NameSpace(17).Items() | Where-Object { $_.Name -eq "Apple iPhone" }
if (-not $dev) { Write-Host "NO PHONE"; exit 1 }
$storage = $dev.GetFolder.Items().Item(0)
$albumItem = $null
foreach ($i in $storage.GetFolder.Items()) { if ($i.Name -eq $AlbumName) { $albumItem = $i } }
if (-not $albumItem) { Write-Host "ALBUM $AlbumName NOT FOUND"; exit 1 }
$src = $albumItem.GetFolder
$destFolder = $sh.NameSpace($Dest)

foreach ($name in $Names) {
  $item = $null
  foreach ($it in $src.Items()) { if ($it.Name -eq $name) { $item = $it } }
  if (-not $item) { Write-Host "MISSING ON PHONE: $name"; continue }

  $out = Join-Path $Dest $name
  if (Test-Path $out) {
    Write-Host ("SKIP {0} (already here, {1:N2} GB)" -f $name, ((Get-Item $out).Length / 1GB))
    continue
  }

  Write-Host "copying $name ..."
  # 16 = yes to all, 512 = no confirmation UI. Returns immediately; the copy
  # continues in the background, which is why the polling below is the point.
  $destFolder.CopyHere($item, 16 -bor 512)

  $stable = 0
  $last = -1
  $waited = 0
  while ($stable -lt 4 -and $waited -lt 3600) {
    Start-Sleep -Seconds 5
    $waited += 5
    if (-not (Test-Path $out)) { continue }
    $size = (Get-Item $out).Length
    if ($size -eq $last -and $size -gt 0) {
      # size held; also confirm nothing still holds it open for writing
      try {
        $fs = [System.IO.File]:Open($out, 'Open', 'Read', 'None')
        $fs.Close()
        $stable++
      } catch { $stable = 0 }
    } else {
      $stable = 0
      if ($size -ne $last) {
        Write-Host ("  {0:N2} GB ..." -f ($size / 1GB))
      }
    }
    $last = $size
  }
  if ($stable -ge 4) {
    Write-Host ("DONE {0}  {1:N2} GB" -f $name, ((Get-Item $out).Length / 1GB))
  } else {
    Write-Host "TIMED OUT waiting for $name to settle"
  }
}

Write-Host "---- all files ----"
Get-ChildItem $Dest -Filter *.MOV | ForEach-Object {
  "{0}  {1:N2} GB" -f $_.Name, ($_.Length / 1GB)
}
