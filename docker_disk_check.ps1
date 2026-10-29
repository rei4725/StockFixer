# docker_disk_check.ps1
# Docker の仮想ディスク (docker_data.vhdx) の肥大化を見張る。
#
# vhdx は中のデータを消しても自動では縮まない（sparse でない）。2026-10-29 に
# 中身 134GB に対して 965GB まで膨らみ、C: の空きが 187GB まで減った。
#
# 想定運用: auto_deploy.ps1 の healthy 確認・掃除の後から呼ばれる（= デプロイ毎に 1 回）。
#   - 毎回 vhdx の大きさを 1 行ログに残す（推移を後から追えるように）
#   - しきい値を超えたら docker-disk-large ラベルの Issue を起票（開いていれば何もしない）
#   - しきい値を下回ったら（= 圧縮された）開いている Issue を自動クローズ
#
# 圧縮は管理者権限と Docker の停止が要るので自動ではやらない。手順は compact_docker_vhdx.ps1。
#
# 単体実行: powershell -ExecutionPolicy Bypass -File C:\src\StockFixer\docker_disk_check.ps1 -DryRun
#
# 注意: $ErrorActionPreference は Stop にしない。
# gh など native コマンドが stderr に出力すると PS 5.1 が
# それを終了エラー扱いして誤検知するため。失敗判定は $LASTEXITCODE で行う。

param(
    # これを超えたら Issue を起票する（GB）
    [double]$ThresholdGB = 300,
    # Issue の起票・クローズをせず、するはずだった操作をログに書くだけにする
    [switch]$DryRun
)

$repoDir   = "C:\src\StockFixer"
$logDir    = Join-Path $repoDir "Logs"
$logFile   = Join-Path $logDir "docker_disk_check.log"
$vhdx      = Join-Path $env:LOCALAPPDATA "Docker\wsl\disk\docker_data.vhdx"
$repo      = "rei4725/StockFixer"
$label     = "docker-disk-large"

function Write-Log($msg) {
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Write-Host $line
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    [System.IO.File]::AppendAllText($logFile, "$line`n", [System.Text.Encoding]::UTF8)
}

function Get-OpenIssues {
    $open = (gh issue list --repo $repo --label $label --state open --json number --jq ".[].number" 2>$null)
    return @($open -split "\r?\n" | Where-Object { $_ })
}

if (-not (Test-Path $vhdx)) {
    Write-Log "[skip] vhdx が見つかりません: $vhdx"
    exit 0
}

$sizeGB = (Get-Item $vhdx).Length / 1GB
$freeGB = (Get-PSDrive C).Free / 1GB
$prefix = if ($DryRun) { "[dry-run] " } else { "" }
Write-Log ("{0}vhdx {1:N1} GB / C: free {2:N1} GB / threshold {3:N0} GB" -f $prefix, $sizeGB, $freeGB, $ThresholdGB)

try {
    $open = Get-OpenIssues

    if ($sizeGB -gt $ThresholdGB) {
        if ($open.Count -gt 0) { exit 0 }   # 起票済み。毎回コメントすると煩いので何もしない

        $title = "🧱 Docker の仮想ディスクが {0:N0} GB に肥大化（しきい値 {1:N0} GB）" -f $sizeGB, $ThresholdGB
        $body = @(
            "## Docker の仮想ディスクの肥大化",
            "",
            "| 項目 | 値 |",
            "|---|---|",
            ("| 検知時刻 | {0} JST |" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
            ("| vhdx | {0:N1} GB |" -f $sizeGB),
            ("| C: の空き | {0:N1} GB |" -f $freeGB),
            ("| しきい値 | {0:N0} GB |" -f $ThresholdGB),
            "",
            "vhdx は中のデータを消しても縮まない。中身の実際の使用量は ``docker system df`` で確かめる。",
            "",
            "### 対処（人間の手作業）",
            "管理者 PowerShell で次を実行する。StockFixer のコンテナが数分止まる。",
            "",
            "``powershell -ExecutionPolicy Bypass -File C:\src\StockFixer\compact_docker_vhdx.ps1``",
            "",
            "圧縮でしきい値を下回れば、次のデプロイでこの Issue は自動でクローズされる。",
            "",
            "> 会社（ai-company）の自動修正の対象ではない（コードの不具合ではなく、手作業の運用のため）。"
        ) -join "`n"

        if ($DryRun) {
            Write-Log "[dry-run] Issue を起票するはずでした: $title"
            exit 0
        }
        try { gh label create $label --repo $repo --color "FBCA04" --description "Docker の仮想ディスクの肥大化（人間が圧縮する）" 2>$null | Out-Null } catch {}
        gh issue create --repo $repo --title $title --label $label --body $body 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Log "[issue] $label の Issue を新規起票" }
        else { Write-Log "[issue] Issue の起票に失敗 (exit=$LASTEXITCODE)" }
    } else {
        foreach ($num in $open) {
            if ($DryRun) {
                Write-Log "[dry-run] #$num をクローズするはずでした"
                continue
            }
            $comment = "✅ vhdx が {0:N1} GB に戻った（しきい値 {1:N0} GB）ため自動クローズします。" -f $sizeGB, $ThresholdGB
            gh issue close $num --repo $repo --comment $comment 2>$null | Out-Null
            Write-Log "[issue] $label #$num を自動クローズ"
        }
    }
} catch {
    Write-Log "[issue] Issue の確認・起票に失敗: $_"
}
exit 0
