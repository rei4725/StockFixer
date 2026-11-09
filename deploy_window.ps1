# deploy_window.ps1
# 自動デプロイを見送る時間帯（発注と約定・夜間の仮の入れ替え・週次のジョブの時間帯）の判定。auto_deploy.ps1 が dot-source する。
#
# 発注のジョブは途中で消えても補完しない（python/run_scheduler.py、#836・#946）ため、
# 8:50 の自動発注・9:05 の約定と 9:55 までのやり直し・16:30 のルールの発注の最中や直前に
# コンテナを起き直すと、その日の決済や買いが出ないまま終わる（#1322）。
# 22:00 の nightly_backtest_promotion（毎日）、火曜 01:00 の週次学習（#1667）、木曜 01:00 の
# Walk-Forward レポートも、起き直しで消える。
#
# 窓の表はデータのファイル deploy_windows.json に置く（このスクリプトはホストの上で動くので、
# 窓を直すだけの変更でコードを触らずに済むようにする）。表が読めない・形が違うときは、
# 安全側に倒してデプロイを見送る。

$DeployWindowsFile = Join-Path $PSScriptRoot "deploy_windows.json"
$script:DeployTimePattern = '^([01][0-9]|2[0-3]):[0-5][0-9]$'
$script:DeployDayNames = @("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")

# 表を読み、窓の配列を返す。読めない・形が違うときは理由の文字列を投げる
function Read-DeployBlackoutWindows([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "deploy_windows.json が無い"
    }
    $data = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    $windows = @($data.windows)
    if ($windows.Count -eq 0) {
        throw "deploy_windows.json に windows が無い"
    }
    foreach ($w in $windows) {
        if ($w.start -notmatch $script:DeployTimePattern) {
            throw "start の形が違う: $($w.start)"
        }
        if ($w.end -ne "24:00" -and $w.end -notmatch $script:DeployTimePattern) {
            throw "end の形が違う: $($w.end)"
        }
        if ($w.end -le $w.start) {
            throw "end が start より後でない: $($w.start)-$($w.end)"
        }
        if (-not ($w.name -is [string]) -or $w.name -eq "") {
            throw "name が無い: $($w.start)-$($w.end)"
        }
        if ($null -ne $w.day -and $script:DeployDayNames -notcontains $w.day) {
            throw "day の形が違う: $($w.day)"
        }
        if ($null -ne $w.daily -and -not ($w.daily -is [bool])) {
            throw "daily は true か false: $($w.daily)"
        }
    }
    return $windows
}

# $now が窓の中なら窓の名前を、外なら $null を返す（start を含み end を含まない）
function Get-DeployBlackout([datetime]$now) {
    try {
        $windows = Read-DeployBlackoutWindows $DeployWindowsFile
    } catch {
        return "窓の表を読めないので見送る（$($_.Exception.Message)）"
    }
    $weekend = $now.DayOfWeek -eq [DayOfWeek]::Saturday -or $now.DayOfWeek -eq [DayOfWeek]::Sunday
    $hm = $now.ToString("HH:mm")
    foreach ($w in $windows) {
        $daily = $w.daily -eq $true
        if ($weekend -and -not $daily) {
            continue
        }
        if ($null -ne $w.day -and $now.DayOfWeek.ToString() -ne $w.day) {
            continue
        }
        if ($hm -ge $w.start -and $hm -lt $w.end) {
            return "$($w.start)-$($w.end) $($w.name)"
        }
    }
    return $null
}
