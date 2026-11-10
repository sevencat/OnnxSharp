# Synthesize a 16 kHz / 16-bit / mono wav with the built-in Chinese voice (Huihui),
# used as the end-to-end ASR test input.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/scripts/tts_huihui.ps1 -Text "..." -Out out.wav
param(
    [string]$Text = "Hello from OnnxSharp.",
    [string]$Out = "huihui.wav"
)

Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$installed = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }
Write-Host "installed voices: $($installed -join ', ')"

$target = $installed | Where-Object { $_ -like "*Huihui*" } | Select-Object -First 1
if (-not $target) { $target = $installed[0] }
$synth.SelectVoice($target)
Write-Host "using voice: $target"

# Paraformer expects 16 kHz mono
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$full = [System.IO.Path]::GetFullPath($Out)
$dir = [System.IO.Path]::GetDirectoryName($full)
if (-not [System.IO.Directory]::Exists($dir)) { [System.IO.Directory]::CreateDirectory($dir) | Out-Null }

$synth.SetOutputToWaveFile($full, $format)
$synth.Speak($Text)
$synth.Dispose()

Write-Host "wrote $full"
