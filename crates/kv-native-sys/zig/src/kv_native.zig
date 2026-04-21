const std = @import("std");
const c = @cImport({
    @cInclude("fcntl.h");
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("sys/stat.h");
    @cInclude("unistd.h");
});

pub const KvStatus = enum(c_int) {
    ok = 0,
    invalid_input = 1,
    invalid_data = 2,
    not_found = 3,
    io = 4,
    oom = 5,
};

pub const KvBuffer = extern struct {
    data: ?[*]u8,
    len: usize,
};

const Fingerprint = [16]u8;

fn sliceFromRaw(ptr: [*]const u8, len: usize) []const u8 {
    return ptr[0..len];
}

fn dupZ(bytes: []const u8) ?[:0]u8 {
    return std.heap.c_allocator.dupeZ(u8, bytes) catch null;
}

fn closeFd(fd: c_int) void {
    _ = c.close(fd);
}

fn fileExists(path_z: [:0]const u8) bool {
    return c.access(path_z.ptr, c.F_OK) == 0;
}

fn openForWrite(path_z: [:0]const u8, out_fd: *c_int) KvStatus {
    const flags: c_int = c.O_WRONLY | c.O_CREAT | c.O_TRUNC;
    const mode: c_uint = 0o644;
    const fd = c.open(path_z.ptr, flags, mode);
    if (fd < 0) return .io;
    out_fd.* = fd;
    return .ok;
}

fn writeAll(fd: c_int, bytes: []const u8) KvStatus {
    var written_total: usize = 0;
    while (written_total < bytes.len) {
        const wrote = c.write(fd, bytes.ptr + written_total, bytes.len - written_total);
        if (wrote <= 0) return .io;
        written_total += @intCast(wrote);
    }
    return .ok;
}

fn writeFileInternal(path: []const u8, bytes: []const u8, atomic: bool) KvStatus {
    if (path.len == 0) return .invalid_input;

    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);

    if (!atomic) {
        var fd: c_int = undefined;
        const open_status = openForWrite(path_z, &fd);
        if (open_status != .ok) return open_status;
        defer closeFd(fd);
        return writeAll(fd, bytes);
    }

    const tmp_path_bytes = std.fmt.allocPrint(std.heap.c_allocator, "{s}.tmp", .{path}) catch {
        return .oom;
    };
    defer std.heap.c_allocator.free(tmp_path_bytes);
    const tmp_path = dupZ(tmp_path_bytes) orelse return .oom;
    defer std.heap.c_allocator.free(tmp_path);
    defer _ = c.unlink(tmp_path.ptr);

    var tmp_fd: c_int = undefined;
    const open_status = openForWrite(tmp_path, &tmp_fd);
    if (open_status != .ok) return open_status;
    defer closeFd(tmp_fd);

    const write_status = writeAll(tmp_fd, bytes);
    if (write_status != .ok) return write_status;

    if (c.rename(tmp_path.ptr, path_z.ptr) != 0) return .io;
    return .ok;
}

fn readFileInternal(path: []const u8, out: *KvBuffer) KvStatus {
    if (path.len == 0) return .invalid_input;
    out.* = .{ .data = null, .len = 0 };

    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);

    if (!fileExists(path_z)) return .not_found;

    const fd = c.open(path_z.ptr, c.O_RDONLY);
    if (fd < 0) return .io;
    defer closeFd(fd);

    var stat_buf: c.struct_stat = undefined;
    if (c.fstat(fd, &stat_buf) != 0) return .io;
    if (stat_buf.st_size < 0) return .invalid_data;

    const capacity: usize = @intCast(stat_buf.st_size);
    if (capacity == 0) return .ok;

    const buffer = std.heap.c_allocator.alloc(u8, capacity) catch return .oom;

    var read_total: usize = 0;
    while (read_total < capacity) {
        const read_now = c.read(fd, buffer.ptr + read_total, capacity - read_total);
        if (read_now < 0) {
            c.free(buffer.ptr);
            return .io;
        }
        if (read_now == 0) break;
        read_total += @intCast(read_now);
    }

    if (read_total == 0) {
        c.free(buffer.ptr);
        return .ok;
    }

    out.* = .{
        .data = buffer.ptr,
        .len = read_total,
    };
    return .ok;
}

fn removeFileInternal(path: []const u8, ignore_not_found: bool) KvStatus {
    if (path.len == 0) return .invalid_input;

    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);

    if (!ignore_not_found and !fileExists(path_z)) return .not_found;
    if (ignore_not_found and !fileExists(path_z)) return .ok;
    if (c.unlink(path_z.ptr) == 0) return .ok;
    if (ignore_not_found and !fileExists(path_z)) return .ok;
    return .io;
}

fn fingerprintFromBytes(bytes: []const u8) ?Fingerprint {
    if (bytes.len != @sizeOf(Fingerprint)) return null;
    var fingerprint: Fingerprint = undefined;
    @memcpy(fingerprint[0..], bytes);
    return fingerprint;
}

fn blockFilenameAlloc(fingerprint: Fingerprint) ?[]u8 {
    const filename = std.heap.c_allocator.alloc(u8, 35) catch return null;
    const hex = "0123456789abcdef";
    for (fingerprint, 0..) |byte, idx| {
        filename[idx * 2] = hex[byte >> 4];
        filename[idx * 2 + 1] = hex[byte & 0x0f];
    }
    filename[32] = '.';
    filename[33] = 'k';
    filename[34] = 'v';
    return filename;
}

fn blockPathAlloc(root: []const u8, fingerprint: Fingerprint) ?[]u8 {
    const filename = blockFilenameAlloc(fingerprint) orelse return null;
    defer std.heap.c_allocator.free(filename);
    return std.fmt.allocPrint(std.heap.c_allocator, "{s}/{s}", .{ root, filename }) catch null;
}

fn writeBlockAtomicInternal(root: []const u8, fingerprint_bytes: []const u8, bytes: []const u8) KvStatus {
    const fingerprint = fingerprintFromBytes(fingerprint_bytes) orelse return .invalid_input;
    const path = blockPathAlloc(root, fingerprint) orelse return .oom;
    defer std.heap.c_allocator.free(path);
    return writeFileInternal(path, bytes, true);
}

fn readBlockInternal(root: []const u8, fingerprint_bytes: []const u8, out: *KvBuffer) KvStatus {
    const fingerprint = fingerprintFromBytes(fingerprint_bytes) orelse return .invalid_input;
    const path = blockPathAlloc(root, fingerprint) orelse return .oom;
    defer std.heap.c_allocator.free(path);
    return readFileInternal(path, out);
}

fn removeBlockInternal(root: []const u8, fingerprint_bytes: []const u8, ignore_not_found: bool) KvStatus {
    const fingerprint = fingerprintFromBytes(fingerprint_bytes) orelse return .invalid_input;
    const path = blockPathAlloc(root, fingerprint) orelse return .oom;
    defer std.heap.c_allocator.free(path);
    return removeFileInternal(path, ignore_not_found);
}

fn blockPathInternal(root: []const u8, fingerprint_bytes: []const u8, out: *KvBuffer) KvStatus {
    const fingerprint = fingerprintFromBytes(fingerprint_bytes) orelse return .invalid_input;
    const path = blockPathAlloc(root, fingerprint) orelse return .oom;
    out.* = .{
        .data = path.ptr,
        .len = path.len,
    };
    return .ok;
}

pub export fn kv_native_write_file(
    path_ptr: [*]const u8,
    path_len: usize,
    bytes_ptr: [*]const u8,
    bytes_len: usize,
) c_int {
    return @intFromEnum(writeFileInternal(
        sliceFromRaw(path_ptr, path_len),
        sliceFromRaw(bytes_ptr, bytes_len),
        false,
    ));
}

pub export fn kv_native_write_file_atomic(
    path_ptr: [*]const u8,
    path_len: usize,
    bytes_ptr: [*]const u8,
    bytes_len: usize,
) c_int {
    return @intFromEnum(writeFileInternal(
        sliceFromRaw(path_ptr, path_len),
        sliceFromRaw(bytes_ptr, bytes_len),
        true,
    ));
}

pub export fn kv_native_read_file(
    path_ptr: [*]const u8,
    path_len: usize,
    out: *KvBuffer,
) c_int {
    return @intFromEnum(readFileInternal(sliceFromRaw(path_ptr, path_len), out));
}

pub export fn kv_native_remove_file(
    path_ptr: [*]const u8,
    path_len: usize,
    ignore_not_found: bool,
) c_int {
    return @intFromEnum(removeFileInternal(
        sliceFromRaw(path_ptr, path_len),
        ignore_not_found,
    ));
}

pub export fn kv_native_block_path(
    root_ptr: [*]const u8,
    root_len: usize,
    fingerprint_ptr: [*]const u8,
    fingerprint_len: usize,
    out: *KvBuffer,
) c_int {
    return @intFromEnum(blockPathInternal(
        sliceFromRaw(root_ptr, root_len),
        sliceFromRaw(fingerprint_ptr, fingerprint_len),
        out,
    ));
}

pub export fn kv_native_write_block_atomic(
    root_ptr: [*]const u8,
    root_len: usize,
    fingerprint_ptr: [*]const u8,
    fingerprint_len: usize,
    bytes_ptr: [*]const u8,
    bytes_len: usize,
) c_int {
    return @intFromEnum(writeBlockAtomicInternal(
        sliceFromRaw(root_ptr, root_len),
        sliceFromRaw(fingerprint_ptr, fingerprint_len),
        sliceFromRaw(bytes_ptr, bytes_len),
    ));
}

pub export fn kv_native_read_block(
    root_ptr: [*]const u8,
    root_len: usize,
    fingerprint_ptr: [*]const u8,
    fingerprint_len: usize,
    out: *KvBuffer,
) c_int {
    return @intFromEnum(readBlockInternal(
        sliceFromRaw(root_ptr, root_len),
        sliceFromRaw(fingerprint_ptr, fingerprint_len),
        out,
    ));
}

pub export fn kv_native_remove_block(
    root_ptr: [*]const u8,
    root_len: usize,
    fingerprint_ptr: [*]const u8,
    fingerprint_len: usize,
    ignore_not_found: bool,
) c_int {
    return @intFromEnum(removeBlockInternal(
        sliceFromRaw(root_ptr, root_len),
        sliceFromRaw(fingerprint_ptr, fingerprint_len),
        ignore_not_found,
    ));
}

pub export fn kv_native_buffer_free(data: ?*u8) void {
    if (data) |ptr| c.free(ptr);
}
