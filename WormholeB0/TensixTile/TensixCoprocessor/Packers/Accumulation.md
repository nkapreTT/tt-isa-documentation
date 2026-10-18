# Accumulation

If a packer's `Pack_L1_Acc` configuration field is set, then rather than overwriting L1, the packer performs `L1 += Datum` for each datum it writes, using the [near-memory accumulate](../../L1.md) functionality of L1. The addition happens in L1's access port, after all of the packer's [format conversions](FormatConversion.md), and so it operates on datums in the final L1 output format.

When a packer [pads](OutputAddressGenerator.md#output-alignment) its final 16 bytes with zeroes, the padding is accumulated as well: every L1 datum covered by the padding has +0 added to it. This leaves most values as they were, but the rules below still apply to them (for example, -0 becomes +0, and denormals become zero).

## Functional model

For the floating-point output formats, the addition is:

```c
uint32_t FlushDenormal(uint32_t x) {
  // Denormals become zero, keeping their sign
  return (x & 0x7f800000) ? x : (x & 0x80000000);
}

uint32_t L1AccumulateFP32(uint32_t L1, uint32_t Datum) {
  float Sum = std::bit_cast<float>(FlushDenormal(L1)) + std::bit_cast<float>(FlushDenormal(Datum)); // IEEE754 addition, rounding to nearest with ties to even
  if (isnan(Sum)) return 0x7fc00000;
  return FlushDenormal(std::bit_cast<uint32_t>(Sum));
}

double BF16ToReal(uint16_t x) {
  bool Sign = x >> 15;
  unsigned Exp = (x >> 7) & 0xff;
  unsigned Man = x & 0x7f;
  double Mag;
  if (Exp == 0) Mag = 0; // Denormals are treated as zero
  else if (Exp == 255 && Man == 0) Mag = INFINITY;
  else Mag = ldexp(128 + Man, Exp - 134); // NB: NaN is treated as a finite value with exponent 255
  return Sign ? -Mag : Mag;
}

uint16_t L1AccumulateBF16(uint16_t L1, uint16_t Datum) {
  if ((L1 == 0x7f80 && Datum == 0xff80) || (L1 == 0xff80 && Datum == 0x7f80)) {
    return 0x7fc0; // +Infinity plus -Infinity gives NaN
  }
  bool L1IsNaN = (L1 & 0x7fff) > 0x7f80;
  if (L1IsNaN) {
    // NaN in L1 behaves as a value larger in magnitude than any Datum other than Infinity
    bool DatumIsInf = (Datum & 0x7fff) == 0x7f80;
    return DatumIsInf ? Datum : ((L1 & 0x8000) | 0x7f80);
  }
  double Sum = BF16ToReal(L1) + BF16ToReal(Datum); // Precise enough that only the rounding below matters
  uint16_t Sign = signbit(Sum) << 15;
  if (fabs(Sum) < 0x1p-126) return Sign; // Flushed to zero before rounding, not after
  if (isinf(Sum)) return Sign | 0x7f80;
  int Exp;
  double Frac = frexp(fabs(Sum), &Exp); // fabs(Sum) == Frac * 2^Exp, 0.5 <= Frac < 1
  uint32_t Man = nearbyint(ldexp(Frac, 8)); // 128 through 256, ties to even
  uint32_t Bits = ((Exp + 126) << 7) + (Man - 128);
  return Sign | (Bits < 0x7f80 ? Bits : 0x7f80);
}

double FP16ToReal(uint16_t x) {
  bool Sign = x >> 15;
  unsigned Exp = (x >> 10) & 0x1f;
  unsigned Man = x & 0x3ff;
  double Mag = Exp ? ldexp(1024 + Man, Exp - 25) : 0; // Denormals are treated as zero
  return Sign ? -Mag : Mag;
}

uint16_t L1AccumulateFP16(uint16_t L1, uint16_t Datum) {
  double Sum = FP16ToReal(L1) + FP16ToReal(Datum); // Exact
  uint16_t Sign = signbit(Sum) << 15;
  if (Sum == 0) return (((L1 >> 10) & 0x1f) >= 30) ? 0x7fff : Sign;
  if (fabs(Sum) < 0x1p-15) return Sign | 0x7fff;
  if (fabs(Sum) < 0x1p-14) return Sign;
  int Exp;
  double Frac = frexp(fabs(Sum), &Exp); // fabs(Sum) == Frac * 2^Exp, 0.5 <= Frac < 1
  uint32_t Man = nearbyint(ldexp(Frac, 11)); // 1024 through 2048, ties to even
  uint32_t Bits = ((Exp + 14) << 10) + (Man - 1024);
  return Sign | (Bits < 0x8000 ? Bits : 0x7fff); // Saturate rather than overflow
}
```

Each datum is accumulated independently of its neighbors. For FP16, every bit pattern with a non-zero exponent is a normal value (there are no bit patterns meaning infinity or NaN; see [FP16 bit patterns](../FloatBitPatterns.md#fp16)), and there are some potentially surprising behaviors:
* If `L1` and `Datum` cancel exactly and the exponent of `L1` is 30 or 31, the result is `0x7FFF` (the largest positive value) rather than +0.
* Results smaller in magnitude than 2<sup>-15</sup> saturate to the largest value of the same sign (`0x7FFF` or `0xFFFF`), whereas results of magnitude at least 2<sup>-15</sup> but less than 2<sup>-14</sup> become zero of the same sign.
