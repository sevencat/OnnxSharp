using OnnxSharp.Ocr.Plate;

namespace OnnxSharp.Tests;

/// <summary>
/// 车牌特化的纯逻辑部分：字符集 / 形态校验 / 受限 CTC 解码。
/// 全部用合成概率，不需要模型文件，因此不 SKIP。
/// </summary>
internal static class PlateOcrTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("plate: charset ranks are unique and position rules hold", CharsetRules);
        yield return new("plate: number parsing covers fuel / new-energy / suffix kinds", NumberParsing);
        yield return new("plate: constrained decode ignores out-of-charset winners", ConstrainedDecode);
        yield return new("plate: beam keeps a 7th char that greedy drops", BeamVersusGreedy);
        yield return new("plate: invalid shape is rejected", RejectsInvalidShape);
    }

    private static void CharsetRules()
    {
        PlateCharset charset = PlateCharset.Default;

        Assert.Equal(31, PlateCharset.Provinces.Length, "province count");
        Assert.Equal(24, PlateCharset.Letters.Length, "letter count (no I/O)");
        Assert.False(charset.Contains('I'), "'I' must not be a plate character");
        Assert.False(charset.Contains('O'), "'O' must not be a plate character");
        Assert.True(charset.Contains('浙') && charset.Contains('A') && charset.Contains('7'), "basic chars");

        // rank 必须与字符一一对应，否则解码出来的文本会错位
        var seen = new HashSet<char>();

        for (int rank = 1; rank <= charset.Count; rank++)
        {
            char c = charset.RankToChar[rank];
            Assert.True(c != '\0', $"rank {rank} maps to NUL");
            Assert.True(seen.Add(c), $"char {c} appears twice");
            Assert.Equal(rank, charset.RankOf(c), $"rank round-trip for {c}");
        }

        // 逐位规则：首位只能是省（或 使/领），第 2 位只能是字母，末位放宽
        Assert.True(Find(charset.AllowedAt(0, 7), '浙') >= 0, "province at position 0");
        Assert.True(Find(charset.AllowedAt(0, 7), 'A') < 0, "letter must not lead");
        Assert.True(Find(charset.AllowedAt(1, 7), 'A') >= 0, "letter at position 1");
        Assert.True(Find(charset.AllowedAt(1, 7), '7') < 0, "digit must not sit at position 1");
        Assert.True(Find(charset.AllowedAt(3, 7), '7') >= 0, "digit in the middle");
        Assert.True(Find(charset.AllowedAt(6, 7), '学') >= 0, "suffix allowed on the last digit");
        Assert.True(Find(charset.AllowedAt(5, 7), '学') < 0, "suffix not allowed in the middle");
    }

    private static void NumberParsing()
    {
        Assert.True(PlateNumber.TryParse("京A12345", out PlateNumber? fuel), "7-digit fuel plate");
        Assert.Equal("京A12345", fuel!.Text, "normalized text");
        Assert.Equal(PlateKind.Fuel, fuel.Kind, "kind");
        Assert.Equal('京', fuel.Province, "province");
        Assert.Equal('A', fuel.Letter, "letter");
        Assert.Equal("12345", fuel.Serial, "serial");

        // 分隔符与全角应被归一化掉
        Assert.True(PlateNumber.TryParse("京A·12345", out PlateNumber? dotted), "dot separator");
        Assert.Equal("京A12345", dotted!.Text, "dot removed");
        Assert.True(PlateNumber.TryParse("粤b12345", out PlateNumber? lower), "lowercase input");
        Assert.Equal("粤B12345", lower!.Text, "uppercased");

        Assert.True(PlateNumber.TryParse("沪AD12345", out PlateNumber? green), "8-digit new energy");
        Assert.Equal(PlateKind.NewEnergy, green!.Kind, "new energy kind");

        Assert.True(PlateNumber.TryParse("京A1234学", out PlateNumber? coach), "coach plate");
        Assert.Equal(PlateKind.Coach, coach!.Kind, "coach kind");

        Assert.False(PlateNumber.TryParse("京I12345", out _, null), "'I' rejected");
        Assert.False(PlateNumber.TryParse("京A1234", out _, null), "6 digits rejected");
        Assert.False(PlateNumber.TryParse("A1234567", out _, null), "must start with a province");
        Assert.False(PlateNumber.TryParse("京112345", out _, null), "digit at letter position");
    }

    private static void ConstrainedDecode()
    {
        // 字典：0 = blank，1 = 浙，2 = A，3 = 1 …；另外放一个「绝不该出现」的汉字
        var dictionary = new List<string> { string.Empty, "浙", "A", "1", "2", "3", "4", "5", "0", "龘" };
        int classes = dictionary.Count;

        // 7 个字 + 首尾各一帧 blank，共 14 帧
        const int steps = 14;
        var probabilities = new float[steps * classes];

        for (int t = 0; t < steps; t++)
        {
            probabilities[t * classes] = 1f;   // 默认整帧都是 blank

            int character = t switch
            {
                1 => 1, 3 => 2, 5 => 3, 7 => 4, 9 => 5, 11 => 6, 13 => 7,
                _ => -1,
            };

            if (character <= 0)
                continue;

            probabilities[t * classes] = 0.05f;
            probabilities[t * classes + character] = 0.95f;

            // 干扰：让「龘」在「A」「1」这两帧上拿到 <b>更高</b>的概率（0.5 > 0.1），
            // 有字符集约束时它必须被忽略
            if (character is 2 or 3)
            {
                probabilities[t * classes] = 0.4f;
                probabilities[t * classes + 9] = 0.5f;
                probabilities[t * classes + character] = 0.1f;
            }
        }

        PlateDecoding decode = PlateDecoder.Decode(probabilities, classes, dictionary);
        Assert.Equal("浙A12345", decode.Text, "constrained decode text");
        Assert.True(decode.IsPlate, "should pass validation");
        Assert.Equal(7, decode.CharScores.Length, "one score per char");
        Assert.True(decode.Score > 0.3f, $"score {decode.Score} too low");
        Assert.Equal(1, decode.ClassIndices[0], "first char maps to 浙");

        // 同样的输入，不做字符集约束时会解出「龘」
        var loose = new PlateDecoderOptions { UseStructure = false, Validate = false };
        PlateDecoding masked = PlateDecoder.Decode(probabilities, classes,
            new[] { string.Empty, "浙", "A", "1", "2", "3", "4", "5", "0", "龘" },
            PlateCharset.Default, loose);

        Assert.True(masked.Text.IndexOf('龘') < 0, $"'{masked.Text}' should not contain 龘");
    }

    private static void BeamVersusGreedy()
    {
        // 构造「第 3 个字只占一帧、且与后一个字同分」的模糊情形：
        // 贪心在 1 帧上容易被邻帧吃掉，beam 应该保住 7 个字
        var dictionary = new List<string> { string.Empty, "京", "B", "6", "7", "8", "9", "0" };
        int classes = dictionary.Count;

        const int steps = 15;
        var probabilities = new float[steps * classes];

        for (int t = 0; t < steps; t++)
            probabilities[t * classes] = 0.98f;

        void Emit(int frame, int character, float probability)
        {
            probabilities[frame * classes] = 1f - probability;
            probabilities[frame * classes + character] = probability;
        }

        Emit(1, 1, 0.9f);    // 京
        Emit(3, 2, 0.85f);   // B
        Emit(5, 3, 0.4f);    // 6（弱）
        Emit(6, 4, 0.45f);   // 7
        Emit(8, 5, 0.8f);    // 8
        Emit(10, 6, 0.8f);   // 9
        Emit(12, 7, 0.8f);   // 0

        var options = new PlateDecoderOptions { BeamWidth = 8, Validate = false };
        PlateDecoding beam = PlateDecoder.Decode(probabilities, classes, dictionary, PlateCharset.Default, options);

        var greedyOptions = new PlateDecoderOptions { BeamWidth = 1, Validate = false };
        PlateDecoding greedy = PlateDecoder.Decode(probabilities, classes, dictionary, PlateCharset.Default, greedyOptions);

        Assert.True(beam.Text.Length >= 7, $"beam produced '{beam.Text}' ({beam.Text.Length} chars)");
        Assert.True(beam.Text.StartsWith("京B", StringComparison.Ordinal), $"beam text '{beam.Text}'");
        Assert.True(beam.Probability >= 0, "probability");
        Assert.True(greedy.Text.Length <= beam.Text.Length, "greedy should not be longer than beam");
    }

    private static void RejectsInvalidShape()
    {
        // 只有 6 个字：带约束解码必须返回空（Validate 默认开）
        var dictionary = new List<string> { string.Empty, "京", "A", "1" };
        int classes = dictionary.Count;

        const int steps = 12;
        var probabilities = new float[steps * classes];

        for (int t = 0; t < steps; t++)
        {
            probabilities[t * classes] = 1f;

            int character = t switch { 1 => 1, 3 => 2, 5 or 7 or 9 or 11 => 3, _ => -1 };
            if (character > 0)
            {
                probabilities[t * classes] = 0.1f;
                probabilities[t * classes + character] = 0.9f;
            }
        }

        PlateDecoding decode = PlateDecoder.Decode(probabilities, classes, dictionary);
        Assert.Equal(string.Empty, decode.Text, "invalid length must be rejected");
        Assert.False(decode.IsPlate, "not a plate");
    }

    private static int Find(ReadOnlySpan<char> span, char value)
    {
        for (int i = 0; i < span.Length; i++)
        {
            if (span[i] == value)
                return i;
        }

        return -1;
    }
}
