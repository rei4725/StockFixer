# deploy_window.ps1
# 自動デプロイを見送る時間帯（発注と約定のジョブの時間帯）の判定。auto_deploy.ps1 が dot-source する。
#
# 発注のジョブは途中で消えても補完しない（python/run_scheduler.py、#836・#946）ため、
# 8:50 の自動発注・9:05 の約定と 9:55 までのやり直し・16:30 のルールの発注の最中や直前に
# コンテナを起き直すと、その日の決済や買いが出ないまま終わる（#1322）。
# 時刻は python/run_scheduler.py の daily_auto_order・daily_settle_orders・daily_rule_signals に合わせる。
# ビルドから up -d まで最長で約6分かかるので、窓の頭はジョブの時刻より 20 分前に置く。

# 平日（月〜金）の窓。時刻は JST（このマシンのローカル時刻）
$DeployBlackoutWindows = @(
    @{ Start = "08:30"; End = "10:00"; Name = "8:50 の自動発注と 9:05 の約定・9:55 までのやり直し" },
    @{ Start = "16:10"; End = "16:50"; Name = "16:30 のルールの発注" }
)

# $now が窓の中なら窓の名前を、外なら $null を返す（Start を含み End を含まない）
function Get-DeployBlackout([datetime]$now) {
    if ($now.DayOfWeek -eq [DayOfWeek]::Saturday -or $now.DayOfWeek -eq [DayOfWeek]::Sunday) {
        return $null
    }
    $hm = $now.ToString("HH:mm")
    foreach ($w in $DeployBlackoutWindows) {
        if ($hm -ge $w.Start -and $hm -lt $w.End) {
            return "$($w.Start)-$($w.End) $($w.Name)"
        }
    }
    return $null
}
