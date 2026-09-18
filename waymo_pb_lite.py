# Stdlib-only reader for Waymo Open Motion Dataset "Scenario" TFRecords.
#
# TFRecord layout (little-endian):
#   uint64 length, uint32 crc(length), byte data[length], uint32 crc(data)
# CRCs are not validated (only the length prefix is needed).
#
# Protobuf wire format: each field is (varint tag) [+ payload], tag =
# (field_number << 3) | wire_type, wire_type in {0: varint, 1: fixed64,
# 2: length-delimited, 5: fixed32}. Unused fields are skipped generically
# by wire type, so only Scenario/Track/ObjectState need to be decoded.
#
# Fields used (numbers from scenario.proto):
#   Scenario: 1 timestamps_seconds, 2 tracks, 5 scenario_id, 6 sdc_track_index
#   Track: 3 states
#   ObjectState: 2 center_x, 3 center_y, 9 velocity_x, 10 velocity_y, 11 valid
import struct


def iter_tfrecords(path):
    """Yield raw serialized-message bytes for each record in a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            len_bytes = f.read(8)
            if len(len_bytes) == 0:
                return
            if len(len_bytes) < 8:
                raise EOFError("Truncated TFRecord length header.")
            (length,) = struct.unpack("<Q", len_bytes)
            f.read(4)
            data = f.read(length)
            if len(data) < length:
                raise EOFError("Truncated TFRecord payload.")
            f.read(4)
            yield data


def _read_varint(buf, pos):
    result = 0
    shift = 0
    n = len(buf)
    while True:
        if pos >= n:
            raise EOFError("Truncated varint.")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def iter_fields(buf):
    """Yield (field_number, wire_type, value) for each top-level field in buf.

    value is: int for wire_type 0 (varint), raw 8-byte bytes for wire_type 1
    (fixed64 -- caller unpacks as double/int64), raw bytes for wire_type 2
    (length-delimited -- string/bytes/embedded message), raw 4-byte bytes
    for wire_type 5 (fixed32 -- caller unpacks as float/int32).
    """
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            val, pos = _read_varint(buf, pos)
            yield field_number, wire_type, val
        elif wire_type == 1:
            val = buf[pos : pos + 8]
            pos += 8
            yield field_number, wire_type, val
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            val = buf[pos : pos + length]
            pos += length
            yield field_number, wire_type, val
        elif wire_type == 5:
            val = buf[pos : pos + 4]
            pos += 4
            yield field_number, wire_type, val
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type} at byte {pos}")


def parse_object_state(buf):
    out = dict(center_x=None, center_y=None, velocity_x=None, velocity_y=None,
               heading=None, valid=None)
    for fn, wt, val in iter_fields(buf):
        if fn == 2 and wt == 1:
            out["center_x"] = struct.unpack("<d", val)[0]
        elif fn == 3 and wt == 1:
            out["center_y"] = struct.unpack("<d", val)[0]
        elif fn == 8 and wt == 5:
            out["heading"] = struct.unpack("<f", val)[0]
        elif fn == 9 and wt == 5:
            out["velocity_x"] = struct.unpack("<f", val)[0]
        elif fn == 10 and wt == 5:
            out["velocity_y"] = struct.unpack("<f", val)[0]
        elif fn == 11 and wt == 0:
            out["valid"] = bool(val)
    return out


def parse_track(buf):
    states = []
    for fn, wt, val in iter_fields(buf):
        if fn == 3 and wt == 2:
            states.append(parse_object_state(val))
    return states


def parse_scenario(buf):
    scenario_id = None
    sdc_track_index = None
    timestamps = []
    tracks = []
    for fn, wt, val in iter_fields(buf):
        if fn == 1 and wt == 1:
            timestamps.append(struct.unpack("<d", val)[0])
        elif fn == 2 and wt == 2:
            tracks.append(parse_track(val))
        elif fn == 5 and wt == 2:
            scenario_id = val.decode("utf-8", errors="replace")
        elif fn == 6 and wt == 0:
            sdc_track_index = val
    return dict(
        scenario_id=scenario_id,
        sdc_track_index=sdc_track_index,
        timestamps_seconds=timestamps,
        tracks=tracks,
    )
