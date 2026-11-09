# deploy_window.ps1
# 自動デプロイを見送る時間帯（発注と約定・夜間の仮の入れ替えのジョブの時間帯）の判定。auto_deploy.ps1 が dot-source する。
#
# 発注のジョブは途中で消えても補完しない（python/run_scheduler.py、#836・#946）ため、
# 8:50 の自動発注・9:05 の約定と 9:55 までのやり直し・16:30 のルールの発注の最中や直前に
# コンテナを起き直すと、その日の決済や買いが出ないまま終わる（#1322）。
# 時刻は python/run_scheduler.py の daily_auto_order・daily_settle_orders・daily_rule_signals に合わせる。
# 22:00 の nightly_backtest_promotion（毎日）は候補の学習と入れ替えの最中に起き直すと
# その夜の評価が消えるため、01:00 までを窓にする。
# ビルドから up -d まで最長で約6分かかるので、窓の頭はジョブの時刻より 20 分前に置く。

# 窓。時刻は JST（このマシンのローカル時刻）。Daily が無い窓は平日（月〜金）だけ。
# 日をまたぐ窓は 24:00 までと 00:00 からの2つに分ける（$hm は "23:59" まで）
# Day がある窓はその曜日だけ。Daily の窓の明けに溜まったデプロイが長いジョブの開始に
# 重ならないよう、火曜 01:00 の週次学習 weekly_model_training（W45 は約21分。上限2で、2回消えると
# その週の学習が抜ける。#1667）と、木曜 01:00 の Walk-Forward レポート（1〜2時間。補完は1回まで）を窓に入れる
$DeployBlackoutWindows = @(
    @{ Start = "08:30"; End = "10:00"; Name = "8:50 の自動発注と 9:05 の約定・9:55 までのやり直し" },
    @{ Start = "16:10"; End = "16:50"; Name = "16:30 のルールの発注" },
    @{ Start = "21:40"; End = "24:00"; Name = "22:00 のバックテストの仮の入れ替え"; Daily = $true },
    @{ Start = "00:00"; End = "01:00"; Name = "22:00 のバックテストの仮の入れ替え"; Daily = $true },
    @{ Start = "01:00"; End = "01:40"; Name = "火曜 01:00 の週次学習"; Day = [DayOfWeek]::Tuesday },
    @{ Start = "01:00"; End = "04:00"; Name = "木曜 01:00 の Walk-Forward レポート"; Day = [DayOfWeek]::Thursday }
)

# $now が窓の中なら窓の名前を、外なら $null を返す（Start を含み End を含まない）
function Get-DeployBlackout([datetime]$now) {
    $weekend = $now.DayOfWeek -eq [DayOfWeek]::Saturday -or $now.DayOfWeek -eq [DayOfWeek]::Sunday
    $hm = $now.ToString("HH:mm")
    foreach ($w in $DeployBlackoutWindows) {
        if ($weekend -and -not $w.Daily) {
            continue
        }
        if ($null -ne $w.Day -and $now.DayOfWeek -ne $w.Day) {
            continue
        }
        if ($hm -ge $w.Start -and $hm -lt $w.End) {
            return "$($w.Start)-$($w.End) $($w.Name)"
        }
    }
    return $null
}
