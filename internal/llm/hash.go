package llm

import "unicode/utf16"

// ShortHash is a fast deterministic hash to shorten long strings. Port of
// utils/hash.ts; like the TS original it iterates UTF-16 code units and uses
// 32-bit multiplication (Math.imul), so hashes match pi's exactly.
func ShortHash(s string) string {
	var h1 uint32 = 0xdeadbeef
	var h2 uint32 = 0x41c6ce57
	for _, ch := range utf16.Encode([]rune(s)) {
		c := uint32(ch)
		h1 = (h1 ^ c) * 2654435761
		h2 = (h2 ^ c) * 1597334677
	}
	h1 = (h1^(h1>>16))*2246822507 ^ (h2^(h2>>13))*3266489909
	h2 = (h2^(h2>>16))*2246822507 ^ (h1^(h1>>13))*3266489909
	return toBase36(h2) + toBase36(h1)
}

// toBase36 renders a uint32 like JS Number.prototype.toString(36).
func toBase36(n uint32) string {
	if n == 0 {
		return "0"
	}
	const digits = "0123456789abcdefghijklmnopqrstuvwxyz"
	var buf [8]byte // 36^7 > 2^32, so 7 chars max
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = digits[n%36]
		n /= 36
	}
	return string(buf[i:])
}
