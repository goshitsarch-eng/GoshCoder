package sessionlog

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"regexp"
	"sync"
	"time"
)

// validSessionID is pi's rule (session-manager.ts assertValidSessionId). A
// session id becomes a filename component, so it is validated rather than
// trusted -- notably when it arrives from -session on the command line.
var validSessionID = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$`)

// ValidateSessionID rejects an id that cannot safely become part of a path.
func ValidateSessionID(id string) error {
	if !validSessionID.MatchString(id) {
		return fmt.Errorf("sessionlog: invalid session id %q: it must be non-empty, contain only letters, digits, '-', '_' and '.', and start and end with a letter or digit", id)
	}
	return nil
}

var uuidv7State struct {
	sync.Mutex
	lastMillis int64
	counter    uint16
}

// NewSessionID returns a uuidv7, matching pi's session ids. Time-ordered ids
// mean a directory listing is already in creation order, which is what makes
// "most recent session for this cwd" a filename comparison rather than a read
// of every header.
//
// Implemented here rather than pulled in as a dependency: it is 30 lines, and
// go.mod's five direct requires are all for the interactive interface.
func NewSessionID() (string, error) {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", fmt.Errorf("sessionlog: generate session id: %w", err)
	}

	millis := time.Now().UnixMilli()
	uuidv7State.Lock()
	switch {
	case millis > uuidv7State.lastMillis:
		uuidv7State.lastMillis = millis
		// Seed the counter from the random block so two ids in the same
		// millisecond stay ordered without being predictable.
		uuidv7State.counter = binary.BigEndian.Uint16(buf[6:8]) & 0x0FFF
	default:
		// Same millisecond, or a clock that went backwards. Keep issuing
		// increasing ids rather than reusing a timestamp.
		millis = uuidv7State.lastMillis
		uuidv7State.counter++
		if uuidv7State.counter > 0x0FFF {
			uuidv7State.lastMillis++
			millis = uuidv7State.lastMillis
			uuidv7State.counter = 0
		}
	}
	counter := uuidv7State.counter
	uuidv7State.Unlock()

	buf[0] = byte(millis >> 40)
	buf[1] = byte(millis >> 32)
	buf[2] = byte(millis >> 24)
	buf[3] = byte(millis >> 16)
	buf[4] = byte(millis >> 8)
	buf[5] = byte(millis)
	buf[6] = 0x70 | byte(counter>>8) // version 7
	buf[7] = byte(counter)
	buf[8] = (buf[8] & 0x3F) | 0x80 // variant 10

	encoded := hex.EncodeToString(buf[:])
	return encoded[0:8] + "-" + encoded[8:12] + "-" + encoded[12:16] + "-" + encoded[16:20] + "-" + encoded[20:32], nil
}

// newEntryID returns an 8-hex-character id that is not already in use, with a
// full 32-character fallback. pi does the same: 8 characters keep a session
// file readable, and the collision check is what makes them safe at that
// length.
func newEntryID(taken func(string) bool) (string, error) {
	var buf [16]byte
	for attempt := 0; attempt < 100; attempt++ {
		if _, err := rand.Read(buf[:4]); err != nil {
			return "", fmt.Errorf("sessionlog: generate entry id: %w", err)
		}
		id := hex.EncodeToString(buf[:4])
		if !taken(id) {
			return id, nil
		}
	}
	if _, err := rand.Read(buf[:]); err != nil {
		return "", fmt.Errorf("sessionlog: generate entry id: %w", err)
	}
	return hex.EncodeToString(buf[:]), nil
}
