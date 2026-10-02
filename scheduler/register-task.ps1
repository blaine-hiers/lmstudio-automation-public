<#
.SYNOPSIS
  Register the load-aware dispatcher (python -m tasks.dispatch) under Windows Task Scheduler.

.DESCRIPTION
  The dispatcher is the ONLY scheduled entry: it runs every 30 minutes, reads
  config\schedule.toml, and runs whatever is due -- only when the PC is quiet
  (see README "Dispatcher"). Settings:
    - every 30 minutes, indefinitely, starting now
    - BelowNormal priority (child tasks inherit it)
    - runs on battery too ("run only if on AC" off)
    - never wakes the PC (no wake timer); a missed trigger just waits
    - IgnoreNew: no second instance while one is still running
    - runs as you, only while you're logged on (the idle/fullscreen checks need
      your desktop session), via pythonw.exe so no console window ever pops up

  Older per-task entries are NOT removed automatically. To list and remove them:
    Get-ScheduledTask -TaskPath "\lmsa\" | Where-Object TaskName -ne "dispatch"
    Get-ScheduledTask -TaskPath "\lmsa\" | Where-Object TaskName -ne "dispatch" |
        Unregister-ScheduledTask -Confirm:$false

.EXAMPLE
  .\scheduler\register-task.ps1            # register (or update) \lmsa\dispatch
  .\scheduler\register-task.ps1 -WhatIf    # show what would be registered
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Name = "dispatch",
    [int]$EveryMinutes = 30
)

$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $py)) { throw "No venv at $py - run the setup in README.md first." }
if (-not (Test-Path (Join-Path $root "tasks\dispatch.py"))) { throw "No tasks\dispatch.py under $root" }

$action = New-ScheduledTaskAction -Execute $py -Argument "-m tasks.dispatch" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -Priority 7 `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6)
# -WakeToRun is deliberately not set: the dispatcher never wakes the PC.
$settings.WakeToRun = $false
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

if ($PSCmdlet.ShouldProcess("\lmsa\$Name", "Register dispatcher every $EveryMinutes min (BelowNormal, no wake, IgnoreNew)")) {
    Register-ScheduledTask -TaskName $Name -TaskPath "\lmsa\" -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal `
        -Description "lmstudio-automation: load-aware dispatcher (python -m tasks.dispatch)" -Force
}
