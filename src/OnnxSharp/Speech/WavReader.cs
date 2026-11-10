using OnnxSharp.Core;

namespace OnnxSharp.Speech;

/// <summary>
/// 极简 WAV 读取（RIFF/PCM）：只支持未压缩的 PCM 8/16/24/32 位与 IEEE float32，
/// 多声道会混成单声道。**不做重采样** —— 采样率不是 16 kHz 时直接报错，
/// 让调用方自己决定用什么重采样器（避免在引擎里塞一个半吊子的重采样）。
/// </summary>
public static class WavReader
{
    /// <summary>读取为单声道 float（范围约 [-1, 1]）。</summary>
    public static (float[] Samples, int SampleRate) Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        if (new string(reader.ReadChars(4)) != "RIFF")
            throw new OnnxSharpException($"{path}: not a RIFF file");
        reader.ReadInt32();                              // riff size
        if (new string(reader.ReadChars(4)) != "WAVE")
            throw new OnnxSharpException($"{path}: not a WAVE file");

        int format = 1;
        int channels = 1;
        int sampleRate = 0;
        int bitsPerSample = 16;

        while (stream.Position + 8 <= stream.Length)
        {
            string id = new(reader.ReadChars(4));
            int size = reader.ReadInt32();
            long next = stream.Position + size + (size & 1);   // chunk 是 2 字节对齐

            if (id == "fmt ")
            {
                format = reader.ReadInt16();
                channels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                reader.ReadInt32();                            // byte rate
                reader.ReadInt16();                            // block align
                bitsPerSample = reader.ReadInt16();

                if (format == 0xFFFE && size >= 40)
                {
                    reader.ReadInt16();                        // cbSize
                    reader.ReadInt16();                        // valid bits
                    reader.ReadInt32();                        // channel mask
                    format = reader.ReadInt16();               // 子格式 GUID 的前两字节
                }
            }
            else if (id == "data")
            {
                if (sampleRate <= 0)
                    throw new OnnxSharpException($"{path}: data chunk before fmt chunk");

                var samples = ReadSamples(reader, size, format, channels, bitsPerSample);
                return (samples, sampleRate);
            }

            stream.Position = next;
        }

        throw new OnnxSharpException($"{path}: no data chunk");
    }

    private static float[] ReadSamples(BinaryReader reader, int byteCount, int format, int channels, int bitsPerSample)
    {
        int bytesPerSample = bitsPerSample / 8;
        if (bytesPerSample <= 0 || channels <= 0)
            throw new OnnxSharpException($"unsupported wav layout (channels={channels}, bits={bitsPerSample})");

        int frames = byteCount / (bytesPerSample * channels);
        var mono = new float[frames];

        for (int frame = 0; frame < frames; frame++)
        {
            float sum = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                float value = (format, bitsPerSample) switch
                {
                    (1, 8) => (reader.ReadByte() - 128) / 128f,
                    (1, 16) => reader.ReadInt16() / 32768f,
                    (1, 24) => ReadInt24(reader) / 8388608f,
                    (1, 32) => reader.ReadInt32() / 2147483648f,
                    (3, 32) => reader.ReadSingle(),
                    _ => throw new OnnxSharpException(
                        $"unsupported wav format (format={format}, bits={bitsPerSample})"),
                };

                sum += value;
            }

            mono[frame] = sum / channels;
        }

        return mono;
    }

    private static int ReadInt24(BinaryReader reader)
    {
        int b0 = reader.ReadByte();
        int b1 = reader.ReadByte();
        int b2 = reader.ReadByte();
        int value = b0 | (b1 << 8) | (b2 << 16);
        return (value & 0x800000) != 0 ? value | ~0xFFFFFF : value;    // 符号扩展
    }
}
