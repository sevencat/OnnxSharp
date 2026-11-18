#!/usr/bin/env pwsh
<#
.SYNOPSIS
    把 Shaders/*.comp 编译成提交入库的 *.spv。

.DESCRIPTION
    设计前提：**build 与 CI 都不需要 Vulkan SDK**。`.spv` 是入库产物，MSBuild 只负责把它们
    作为 EmbeddedResource 打进程序集；只有在改动 `.comp` 之后才需要跑这个脚本。

    编译器按顺序查找：
      $env:VULKAN_SDK\Bin\glslc.exe → PATH 里的 glslc → PATH 里的 glslang
      → 仓库根 bin\glslang.exe（便携版，不入库）

    增量规则：`.spv` 比 `.comp` 新就跳过；`-Force` 强制重编。

.EXAMPLE
    pwsh src/OnnxSharp.Vulkan/build-shaders.ps1
    pwsh src/OnnxSharp.Vulkan/build-shaders.ps1 -Force

.NOTES
    本文件必须保持 UTF-8 **带 BOM**。Windows PowerShell 5.1 会把无 BOM 的脚本按系统
    ANSI 代码页读取，中文注释会被解坏并触发语法错误（实测报「string is missing the terminator」）。
    没有 pwsh 的机器用 `powershell -ExecutionPolicy Bypass -File` 跑。
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$shaderDir = Join-Path $PSScriptRoot 'Shaders'
if (-not (Test-Path $shaderDir)) {
    throw "找不到 shader 目录: $shaderDir"
}

function Resolve-ShaderCompiler {
    $candidates = @()
    if ($env:VULKAN_SDK) {
        $candidates += (Join-Path $env:VULKAN_SDK 'Bin\glslc.exe')
        $candidates += (Join-Path $env:VULKAN_SDK 'Bin\glslangValidator.exe')
    }
    foreach ($n in 'glslc', 'glslang', 'glslangValidator') {
        $cmd = Get-Command $n -ErrorAction SilentlyContinue
        if ($cmd) { $candidates += $cmd.Source }
    }
    $candidates += (Join-Path $PSScriptRoot '..\..\bin\glslang.exe')

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
    }
    throw @'
找不到 shader 编译器。请任选其一：
  - 安装 Vulkan SDK 并设置 VULKAN_SDK
  - 把 glslang/glslc 放进 PATH
  - 把 glslang.exe 放到仓库根 bin\ 目录（bin/ 已在 .gitignore 中）
'@
}

$compiler = Resolve-ShaderCompiler
$isGlslc = [IO.Path]::GetFileNameWithoutExtension($compiler) -eq 'glslc'
Write-Host "compiler : $compiler"
Write-Host ("mode     : {0}  target=vulkan1.1" -f $(if ($isGlslc) { 'glslc -O' } else { 'glslang -Os' }))

$sources = @(Get-ChildItem -Path $shaderDir -Filter *.comp -File | Sort-Object Name)
if ($sources.Count -eq 0) {
    Write-Host '没有 .comp 源文件，结束。'
    exit 0
}

$built = 0
$skipped = 0
foreach ($src in $sources) {
    $out = [IO.Path]::ChangeExtension($src.FullName, '.spv')

    if (-not $Force -and (Test-Path $out) -and
        (Get-Item $out).LastWriteTimeUtc -ge $src.LastWriteTimeUtc) {
        $skipped++
        continue
    }

    if ($isGlslc) {
        $compilerArgs = @('-O', '--target-env=vulkan1.1', '-o', $out, $src.FullName)
    }
    else {
        # glslang 把输入文件名打到 stderr，不能当成错误；只看退出码
        $compilerArgs = @('-V', '-Os', '--target-env', 'vulkan1.1', '-o', $out, $src.FullName)
    }

    # glslang 会把输入文件名写到 stderr，在 $ErrorActionPreference='Stop' 下会被当成
    # 终止性错误（NativeCommandError），所以这里临时放宽，只认退出码
    $prevEa = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $log = & $compiler @compilerArgs 2>&1 | Out-String
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prevEa

    if ($exit -ne 0) {
        Write-Host $log
        throw "编译失败: $($src.Name)"
    }

    # 校验产物：长度是 4 的倍数且 magic = 0x07230203，避免把空文件或文本混进资源
    $bytes = [IO.File]::ReadAllBytes($out)
    if ($bytes.Length -lt 20 -or ($bytes.Length % 4) -ne 0) {
        throw "$($src.Name): .spv 长度非法 ($($bytes.Length) B)"
    }
    $magic = [BitConverter]::ToUInt32($bytes, 0)
    if ($magic -ne 0x07230203) {
        throw ("$($src.Name): SPIR-V magic 错误 0x{0:X8}" -f $magic)
    }

    Write-Host ("  {0} -> {1} ({2} B)" -f $src.Name, [IO.Path]::GetFileName($out), $bytes.Length)
    $built++
}

Write-Host "done     : built=$built  up-to-date=$skipped"
