# What is actually on the phone right now, newest first.
#
# pull-footage.ps1 hardcodes an album and two filenames from a previous
# transfer. Discovering them instead means a new session's videos can be found
# without guessing, and it makes the "which two are the latest" question
# answerable from the device rather than from memory.
$ErrorActionPreference = "Stop"
$sh = New-Object -ComObject Shell.Application
$dev = $sh.NameSpace(17).Items() | Where-Object { $_.Name -eq "Apple iPhone" }
if (-not $dev) { Write-Host "NO PHONE - is it unlocked and Trusted?"; exit 1 }
$storage = $dev.GetFolder.Items().Item(0)
$rows = @()
foreach ($album in $storage.GetFolder.Items()) {
  if (-not $album.IsFolder) { continue }
  foreach ($f in $album.GetFolder.Items()) {
    if ($f.Name -notmatch '\.(MOV|MP4)$') { continue }
    $rows += [pscustomobject]@{
      Album = $album.Name
      Name  = $f.Name
      Size  = $album.GetFolder.GetDetailsOf($f, 1)
      Date  = $album.GetFolder.GetDetailsOf($f, 3)
    }
  }
}
$rows | Sort-Object Date -Descending | Select-Object -First 14 |
  ForEach-Object { '{0,-10} {1,-16} {2,12} {3}' -f $_.Album, $_.Name, $_.Size, $_.Date }
Write-Host ""
Write-Host ("total videos on device: " + $rows.Count)
