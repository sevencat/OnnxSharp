// CLI 用的最小图像解码器：只认 PPM(P6) 和未压缩 BMP。
// 不引第三方包，也不进产品库 —— 引擎侧只吃「像素 + 宽高」，谁解码由调用方决定。
using System.Buffers.Binary;

/// <summary>解出来的紧凑三通道像素（RGB24，每行 <c>Width * 3</c> 字节）。</summary>
internal readonly record struct ImageData(byte[] Pixels, int Width, int Height)
{
    public int RowStride => Width * 3;
}

internal static class ImageFile
{
    public static ImageData Load(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);

        if (bytes.Length >= 2 && bytes[0] == (byte)'B' && bytes[1] == (byte)'M')
            return LoadBmp(bytes);

        if (bytes.Length >= 2 && bytes[0] == (byte)'P' && bytes[1] == (byte)'6')
            return LoadPpm(bytes);

        throw new InvalidDataException(
            $"{path}: unsupported image format (PPM P6 or uncompressed BMP only)");
    }

    private static ImageData LoadPpm(byte[] bytes)
    {
        int position = 2;

        int NextToken()
        {
            while (position < bytes.Length)
            {
                byte current = bytes[position];
                if (current is (byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n')
                {
                    position++;
                    continue;
                }

                if (current == (byte)'#')
                {
                    while (position < bytes.Length && bytes[position] != (byte)'\n')
                        position++;
                    continue;
                }

                break;
            }

            int start = position;
            while (position < bytes.Length && bytes[position] >= (byte)'0' && bytes[position] <= (byte)'9')
                position++;

            if (start == position)
                throw new InvalidDataException("PPM: malformed header");

            return int.Parse(bytes.AsSpan(start, position - start));
        }

        int width = NextToken();
        int height = NextToken();
        int maxValue = NextToken();

        if (maxValue != 255)
            throw new InvalidDataException($"PPM: max value {maxValue} is not supported (need 255)");

        if (width <= 0 || height <= 0)
            throw new InvalidDataException($"PPM: invalid size {width}x{height}");

        // 头和数据之间只允许一个空白字符。
        position++;

        int expected = checked(width * height * 3);
        if (bytes.Length - position < expected)
            throw new InvalidDataException(
                $"PPM: truncated pixel data ({bytes.Length - position} bytes, need {expected})");

        var pixels = new byte[expected];
        Buffer.BlockCopy(bytes, position, pixels, 0, expected);
        return new ImageData(pixels, width, height);
    }

    private static ImageData LoadBmp(byte[] bytes)
    {
        if (bytes.Length < 54)
            throw new InvalidDataException("BMP: file is too short");

        int pixelOffset = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(10));
        int headerSize = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(14));
        if (headerSize < 40)
            throw new InvalidDataException($"BMP: unsupported header size {headerSize} (need BITMAPINFOHEADER)");

        int width = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(18));
        int rawHeight = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(22));
        int bitsPerPixel = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(28));
        int compression = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(30));

        if (compression != 0)
            throw new InvalidDataException($"BMP: compression {compression} is not supported (uncompressed only)");

        if (bitsPerPixel is not (24 or 32))
            throw new InvalidDataException($"BMP: {bitsPerPixel}bpp is not supported (24 or 32)");

        // height 为负表示自顶向下存储。
        bool bottomUp = rawHeight > 0;
        int height = Math.Abs(rawHeight);
        if (width <= 0 || height <= 0)
            throw new InvalidDataException($"BMP: invalid size {width}x{height}");

        int bytesPerPixel = bitsPerPixel / 8;
        int stride = (width * bitsPerPixel + 31) / 32 * 4;
        if (bytes.Length < pixelOffset + (long)stride * height)
            throw new InvalidDataException("BMP: truncated pixel data");

        var pixels = new byte[checked(width * height * 3)];

        for (int y = 0; y < height; y++)
        {
            int sourceRow = bottomUp ? height - 1 - y : y;
            int source = pixelOffset + sourceRow * stride;
            int destination = y * width * 3;

            for (int x = 0; x < width; x++)
            {
                // BMP 存的是 BGR(A)。
                int pixel = source + x * bytesPerPixel;
                pixels[destination + x * 3] = bytes[pixel + 2];
                pixels[destination + x * 3 + 1] = bytes[pixel + 1];
                pixels[destination + x * 3 + 2] = bytes[pixel];
            }
        }

        return new ImageData(pixels, width, height);
    }
}
