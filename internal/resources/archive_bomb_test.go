package resources

// A sparse-file tar bomb, and the bound that stops it.
//
// The archive built here is roughly 200 bytes and declares a member of 1 GiB.
// GNU sparse encoding stores only the non-hole ranges; archive/tar materializes
// the holes as zeroes on read. Before the fix this reached io.ReadAll and
// allocated over two gigabytes before any size check downstream could run --
// from a file small enough to arrive in an email.
//
// The counter on the gzip stream does not catch it, and cannot: the expansion
// happens after decompression, so the bytes are never read from the compressed
// stream at all. That is why the member read itself is bounded rather than its
// length being checked afterwards.

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"runtime"
	"testing"
)

// writeOctal writes a NUL-terminated octal field, as the ustar header format
// requires.
func writeOctal(field []byte, value int64) {
	copy(field, fmt.Sprintf("%0*o", len(field)-1, value))
	field[len(field)-1] = 0
}

// ustarHeader builds one 512-byte header block with a valid checksum.
func ustarHeader(name string, size int64, typeflag byte) []byte {
	block := make([]byte, 512)
	copy(block[0:100], name)
	writeOctal(block[100:108], 0o600)
	writeOctal(block[108:116], 0)
	writeOctal(block[116:124], 0)
	writeOctal(block[124:136], size)
	writeOctal(block[136:148], 0)
	block[156] = typeflag
	copy(block[257:263], "ustar\x00")
	copy(block[263:265], "00")

	// The checksum is computed with its own field read as spaces.
	for i := 148; i < 156; i++ {
		block[i] = ' '
	}
	var sum int64
	for _, b := range block {
		sum += int64(b)
	}
	writeOctal(block[148:155], sum)
	block[155] = ' '
	return block
}

// padToBlock rounds the buffer up to a 512-byte boundary.
func padToBlock(buf *bytes.Buffer) {
	if remainder := buf.Len() % 512; remainder != 0 {
		buf.Write(make([]byte, 512-remainder))
	}
}

// paxRecord formats one PAX extended-header record, whose declared length
// includes the digits of the length itself.
func paxRecord(key, value string) string {
	length := len(key) + len(value) + 3 // ' ', '=', '\n'
	for {
		total := length + len(fmt.Sprint(length))
		if len(fmt.Sprint(total)) == len(fmt.Sprint(length)) {
			length = total
			break
		}
		length = total
	}
	return fmt.Sprintf("%d %s=%s\n", length, key, value)
}

// buildSparseBomb produces a tiny archive whose one member claims logicalSize
// bytes while carrying a single byte of real data.
func buildSparseBomb(logicalSize int64) []byte {
	var archive bytes.Buffer

	records := paxRecord("GNU.sparse.major", "0") +
		paxRecord("GNU.sparse.minor", "1") +
		paxRecord("GNU.sparse.name", "user/bomb.md") +
		paxRecord("GNU.sparse.size", fmt.Sprint(logicalSize)) +
		paxRecord("GNU.sparse.numblocks", "1") +
		paxRecord("GNU.sparse.map", "0,1")

	archive.Write(ustarHeader("PaxHeaders/bomb", int64(len(records)), 'x'))
	archive.WriteString(records)
	padToBlock(&archive)

	archive.Write(ustarHeader("bomb", 1, '0'))
	archive.WriteByte('A')
	padToBlock(&archive)

	archive.Write(make([]byte, 1024)) // two zero blocks end the archive

	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	writer.Write(archive.Bytes())
	writer.Close()
	return compressed.Bytes()
}

// TestSparseMemberCannotAllocateBeyondTheLimit is the regression. It asserts
// allocation, not just the returned warning: the old code also reported the
// member as skipped -- after it had already allocated the two gigabytes.
func TestSparseMemberCannotAllocateBeyondTheLimit(t *testing.T) {
	const logical = 1 << 30 // 1 GiB claimed, ~200 bytes on the wire
	archive := buildSparseBomb(logical)

	if len(archive) > 4096 {
		t.Fatalf("the bomb is %d bytes; it is meant to be tiny", len(archive))
	}

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	prompts, _, warnings, err := ReadArchive(bytes.NewReader(archive))
	runtime.ReadMemStats(&after)

	if err != nil {
		t.Fatalf("ReadArchive returned an error rather than skipping the member: %v", err)
	}
	if len(prompts) != 0 {
		t.Fatalf("the oversized member was restored: %+v", prompts)
	}
	if len(warnings) == 0 {
		t.Fatal("the member was dropped without saying so")
	}

	allocated := (after.TotalAlloc - before.TotalAlloc) >> 20
	// Generous: the point is the difference between a few MiB and 2,000.
	if allocated > 64 {
		t.Fatalf("reading a %d-byte archive allocated %d MiB", len(archive), allocated)
	}
}

// TestHonestlySizedMemberStillLoads is the negative control. A bound that also
// rejected ordinary prompts would pass the test above for the wrong reason.
func TestHonestlySizedMemberStillLoads(t *testing.T) {
	archive := writeArchiveOf(t, []ArchivedPrompt{
		{Name: "ordinary", Target: SaveUser, Body: "a perfectly normal prompt"},
	})

	prompts, _, warnings, err := ReadArchive(bytes.NewReader(archive))
	if err != nil {
		t.Fatalf("ReadArchive: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings for a well-formed archive: %v", warnings)
	}
	if len(prompts) != 1 || prompts[0].Body != "a perfectly normal prompt" {
		t.Fatalf("prompts = %+v", prompts)
	}
}

// TestMemberLargerThanTheLimitIsSkippedNotFatal keeps one oversized prompt from
// costing the whole archive.
func TestMemberLargerThanTheLimitIsSkippedNotFatal(t *testing.T) {
	archive := writeArchiveOf(t, []ArchivedPrompt{
		{Name: "huge", Target: SaveUser, Body: string(bytes.Repeat([]byte("x"), maxResourceBytes+1))},
		{Name: "small", Target: SaveUser, Body: "kept"},
	})

	prompts, _, warnings, err := ReadArchive(bytes.NewReader(archive))
	if err != nil {
		t.Fatalf("ReadArchive: %v", err)
	}
	if len(prompts) != 1 || prompts[0].Name != "small" {
		t.Fatalf("prompts = %+v, want only the small one", prompts)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one naming the skipped member", warnings)
	}
}
