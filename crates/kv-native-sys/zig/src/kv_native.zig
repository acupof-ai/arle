const std = @import("std");
const c = @cImport({
    @cInclude("fcntl.h");
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("sys/mman.h");
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
const wal_magic = "KVWAL001";
const shm_magic = "KVSHM001";
const mmap_path_cap = 512;
const shm_name_cap = 128;
const map_anon_flag: c_int = if (@hasDecl(c, "MAP_ANON")) c.MAP_ANON else c.MAP_ANONYMOUS;

pub const KvMmapDescriptor = extern struct {
    len: usize,
    path_len: usize,
    path: [mmap_path_cap]u8,
};

pub const KvSharedMemoryDescriptor = extern struct {
    len: usize,
    generation: u64,
    name_len: usize,
    name: [shm_name_cap]u8,
};

pub const KvHostArenaRegion = extern struct {
    offset: u64,
    len: usize,
};

const ShmHeader = extern struct {
    magic: [shm_magic.len]u8,
    generation: u64,
    payload_len: u64,
};

const KvHostArena = struct {
    mapping: ?*anyopaque,
    capacity_bytes: usize,
    next_offset: usize,
    free_list: std.ArrayListUnmanaged(KvHostArenaRegion),
};

var shm_generation_counter = std.atomic.Value(u64).init(1);

fn sliceFromRaw(ptr: [*]const u8, len: usize) []const u8 {
    return ptr[0..len];
}

fn dupZ(bytes: []const u8) ?[:0]u8 {
    return std.heap.c_allocator.dupeZ(u8, bytes) catch null;
}

fn closeFd(fd: c_int) void {
    _ = c.close(fd);
}

fn syncFd(fd: c_int) KvStatus {
    if (c.fsync(fd) != 0) return .io;
    return .ok;
}

fn mapFd(fd: c_int, len: usize, prot: c_int) ?*anyopaque {
    return c.mmap(null, len, prot, c.MAP_SHARED, fd, 0);
}

fn unmapAddr(addr: ?*anyopaque, len: usize) void {
    _ = c.munmap(addr, len);
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

fn parentDirAlloc(path: []const u8) ?[:0]u8 {
    const idx = std.mem.lastIndexOfScalar(u8, path, '/') orelse {
        return std.heap.c_allocator.dupeZ(u8, ".") catch null;
    };
    if (idx == 0) {
        return std.heap.c_allocator.dupeZ(u8, "/") catch null;
    }
    return std.heap.c_allocator.dupeZ(u8, path[0..idx]) catch null;
}

fn fsyncParentDir(path: []const u8) KvStatus {
    const parent_z = parentDirAlloc(path) orelse return .oom;
    defer std.heap.c_allocator.free(parent_z);

    const flags: c_int = c.O_RDONLY | c.O_DIRECTORY;
    const fd = c.open(parent_z.ptr, flags, @as(c_uint, 0));
    if (fd < 0) return .io;
    defer closeFd(fd);
    return syncFd(fd);
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
    const data_sync_status = syncFd(tmp_fd);
    if (data_sync_status != .ok) return data_sync_status;

    if (c.rename(tmp_path.ptr, path_z.ptr) != 0) return .io;
    return fsyncParentDir(path);
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

fn ensurePathFits(bytes: []const u8, cap: usize) bool {
    return bytes.len <= cap;
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

fn walAppendInternal(path: []const u8, kind: u8, key: []const u8, value: []const u8) KvStatus {
    if (key.len > std.math.maxInt(u32) or value.len > std.math.maxInt(u32)) return .invalid_input;

    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);

    const flags: c_int = c.O_WRONLY | c.O_CREAT | c.O_APPEND;
    const mode: c_uint = 0o644;
    const fd = c.open(path_z.ptr, flags, mode);
    if (fd < 0) return .io;
    defer closeFd(fd);

    var stat_buf: c.struct_stat = undefined;
    if (c.fstat(fd, &stat_buf) != 0) return .io;
    if (stat_buf.st_size == 0) {
        if (writeAll(fd, wal_magic) != .ok) return .io;
    }

    var header: [9]u8 = undefined;
    header[0] = kind;
    std.mem.writeInt(u32, header[1..5], @intCast(key.len), .little);
    std.mem.writeInt(u32, header[5..9], @intCast(value.len), .little);
    if (writeAll(fd, &header) != .ok) return .io;
    if (writeAll(fd, key) != .ok) return .io;
    if (writeAll(fd, value) != .ok) return .io;
    if (c.fsync(fd) != 0) return .io;
    return .ok;
}

fn walReplayInternal(path: []const u8, out: *KvBuffer) KvStatus {
    const status = readFileInternal(path, out);
    if (status == .not_found) {
        out.* = .{ .data = null, .len = 0 };
        return .ok;
    }
    if (status != .ok) return status;
    if (out.len == 0) return .ok;

    const bytes = out.data.?[0..out.len];
    if (bytes.len < wal_magic.len or !std.mem.eql(u8, bytes[0..wal_magic.len], wal_magic)) {
        c.free(out.data);
        out.* = .{ .data = null, .len = 0 };
        return .invalid_data;
    }

    var cursor: usize = wal_magic.len;
    while (cursor < bytes.len) {
        if (cursor + 9 > bytes.len) {
            c.free(out.data);
            out.* = .{ .data = null, .len = 0 };
            return .invalid_data;
        }
        const key_len = std.mem.readInt(
            u32,
            @ptrCast(bytes[cursor + 1 .. cursor + 5]),
            .little,
        );
        const value_len = std.mem.readInt(
            u32,
            @ptrCast(bytes[cursor + 5 .. cursor + 9]),
            .little,
        );
        cursor += 9;
        const record_len: usize = @as(usize, key_len) + @as(usize, value_len);
        if (cursor + record_len > bytes.len) {
            c.free(out.data);
            out.* = .{ .data = null, .len = 0 };
            return .invalid_data;
        }
        cursor += record_len;
    }
    return .ok;
}

fn mmapCreateInternal(path: []const u8, len: usize, out: *KvMmapDescriptor) KvStatus {
    if (!ensurePathFits(path, mmap_path_cap)) return .invalid_input;
    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);

    const flags: c_int = c.O_RDWR | c.O_CREAT | c.O_TRUNC;
    const mode: c_uint = 0o644;
    const fd = c.open(path_z.ptr, flags, mode);
    if (fd < 0) return .io;
    defer closeFd(fd);
    if (c.ftruncate(fd, @intCast(len)) != 0) return .io;

    out.* = std.mem.zeroes(KvMmapDescriptor);
    out.len = len;
    out.path_len = path.len;
    @memcpy(out.path[0..path.len], path);
    return .ok;
}

fn descriptorPath(desc: *const KvMmapDescriptor) ?[]const u8 {
    if (desc.path_len > desc.path.len) return null;
    return desc.path[0..desc.path_len];
}

fn mmapWriteInternal(desc: *const KvMmapDescriptor, offset: usize, bytes: []const u8) KvStatus {
    if (offset + bytes.len > desc.len) return .invalid_input;
    const path = descriptorPath(desc) orelse return .invalid_input;
    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);
    const fd = c.open(path_z.ptr, c.O_RDWR);
    if (fd < 0) return .io;
    defer closeFd(fd);
    const mapping = mapFd(fd, desc.len, c.PROT_READ | c.PROT_WRITE);
    if (mapping == c.MAP_FAILED) return .io;
    defer unmapAddr(mapping, desc.len);
    const base: [*]u8 = @ptrCast(@alignCast(mapping));
    @memcpy(base[offset .. offset + bytes.len], bytes);
    if (c.msync(mapping, desc.len, c.MS_SYNC) != 0) return .io;
    return .ok;
}

fn mmapReadInternal(desc: *const KvMmapDescriptor, offset: usize, bytes_len: usize, out: *KvBuffer) KvStatus {
    if (offset + bytes_len > desc.len) return .invalid_input;
    const path = descriptorPath(desc) orelse return .invalid_input;
    const path_z = dupZ(path) orelse return .oom;
    defer std.heap.c_allocator.free(path_z);
    const fd = c.open(path_z.ptr, c.O_RDONLY);
    if (fd < 0) return .io;
    defer closeFd(fd);
    const mapping = mapFd(fd, desc.len, c.PROT_READ);
    if (mapping == c.MAP_FAILED) return .io;
    defer unmapAddr(mapping, desc.len);

    const buffer = std.heap.c_allocator.alloc(u8, bytes_len) catch return .oom;
    const base: [*]u8 = @ptrCast(@alignCast(mapping));
    @memcpy(buffer[0..bytes_len], base[offset .. offset + bytes_len]);
    out.* = .{ .data = buffer.ptr, .len = bytes_len };
    return .ok;
}

fn shmNameFromDescriptor(desc: *const KvSharedMemoryDescriptor) ?[]const u8 {
    if (desc.name_len > desc.name.len) return null;
    return desc.name[0..desc.name_len];
}

fn shmTotalLen(payload_len: usize) ?usize {
    return std.math.add(usize, @sizeOf(ShmHeader), payload_len) catch null;
}

fn nextShmGeneration() u64 {
    const generation = shm_generation_counter.fetchAdd(1, .monotonic);
    return if (generation == 0) shm_generation_counter.fetchAdd(1, .monotonic) else generation;
}

fn shmHeaderPtr(mapping: ?*anyopaque) *ShmHeader {
    return @ptrCast(@alignCast(mapping));
}

fn shmPayloadBase(mapping: ?*anyopaque) [*]u8 {
    const base: [*]u8 = @ptrCast(@alignCast(mapping));
    return base + @sizeOf(ShmHeader);
}

fn validateShmHeader(desc: *const KvSharedMemoryDescriptor, header: *const ShmHeader) KvStatus {
    if (!std.mem.eql(u8, header.magic[0..], shm_magic)) return .invalid_data;
    if (header.generation != desc.generation) return .invalid_data;
    if (header.payload_len != desc.len) return .invalid_data;
    return .ok;
}

fn shmCreateInternal(name: []const u8, len: usize, out: *KvSharedMemoryDescriptor) KvStatus {
    if (name.len == 0 or name[0] != '/' or !ensurePathFits(name, shm_name_cap)) return .invalid_input;
    const total_len = shmTotalLen(len) orelse return .invalid_input;
    const name_z = dupZ(name) orelse return .oom;
    defer std.heap.c_allocator.free(name_z);

    const fd = c.shm_open(name_z.ptr, c.O_RDWR | c.O_CREAT | c.O_EXCL, @as(c_uint, 0o600));
    if (fd < 0) return .io;
    defer closeFd(fd);
    errdefer _ = c.shm_unlink(name_z.ptr);
    if (c.ftruncate(fd, @intCast(total_len)) != 0) return .io;
    const mapping = mapFd(fd, total_len, c.PROT_READ | c.PROT_WRITE);
    if (mapping == c.MAP_FAILED) return .io;
    defer unmapAddr(mapping, total_len);

    const generation = nextShmGeneration();
    const header = shmHeaderPtr(mapping);
    header.* = std.mem.zeroes(ShmHeader);
    @memcpy(header.magic[0..], shm_magic);
    header.generation = generation;
    header.payload_len = len;

    out.* = std.mem.zeroes(KvSharedMemoryDescriptor);
    out.len = len;
    out.generation = generation;
    out.name_len = name.len;
    @memcpy(out.name[0..name.len], name);
    return .ok;
}

fn shmWriteInternal(desc: *const KvSharedMemoryDescriptor, offset: usize, bytes: []const u8) KvStatus {
    if (offset + bytes.len > desc.len) return .invalid_input;
    const name = shmNameFromDescriptor(desc) orelse return .invalid_input;
    const total_len = shmTotalLen(desc.len) orelse return .invalid_input;
    const name_z = dupZ(name) orelse return .oom;
    defer std.heap.c_allocator.free(name_z);
    const fd = c.shm_open(name_z.ptr, c.O_RDWR, @as(c_uint, 0o600));
    if (fd < 0) return .io;
    defer closeFd(fd);
    const mapping = mapFd(fd, total_len, c.PROT_READ | c.PROT_WRITE);
    if (mapping == c.MAP_FAILED) return .io;
    defer unmapAddr(mapping, total_len);
    const header = shmHeaderPtr(mapping);
    const header_status = validateShmHeader(desc, header);
    if (header_status != .ok) return header_status;
    const payload = shmPayloadBase(mapping);
    @memcpy(payload[offset .. offset + bytes.len], bytes);
    if (c.msync(mapping, total_len, c.MS_SYNC) != 0) return .io;
    return .ok;
}

fn shmReadInternal(desc: *const KvSharedMemoryDescriptor, offset: usize, bytes_len: usize, out: *KvBuffer) KvStatus {
    if (offset + bytes_len > desc.len) return .invalid_input;
    const name = shmNameFromDescriptor(desc) orelse return .invalid_input;
    const total_len = shmTotalLen(desc.len) orelse return .invalid_input;
    const name_z = dupZ(name) orelse return .oom;
    defer std.heap.c_allocator.free(name_z);
    const fd = c.shm_open(name_z.ptr, c.O_RDONLY, @as(c_uint, 0o600));
    if (fd < 0) return .io;
    defer closeFd(fd);
    const mapping = mapFd(fd, total_len, c.PROT_READ);
    if (mapping == c.MAP_FAILED) return .io;
    defer unmapAddr(mapping, total_len);
    const header = shmHeaderPtr(mapping);
    const header_status = validateShmHeader(desc, header);
    if (header_status != .ok) return header_status;
    const buffer = std.heap.c_allocator.alloc(u8, bytes_len) catch return .oom;
    const payload = shmPayloadBase(mapping);
    @memcpy(buffer[0..bytes_len], payload[offset .. offset + bytes_len]);
    out.* = .{ .data = buffer.ptr, .len = bytes_len };
    return .ok;
}

fn shmUnlinkInternal(desc: *const KvSharedMemoryDescriptor) KvStatus {
    const name = shmNameFromDescriptor(desc) orelse return .invalid_input;
    const total_len = shmTotalLen(desc.len) orelse return .invalid_input;
    const name_z = dupZ(name) orelse return .oom;
    defer std.heap.c_allocator.free(name_z);
    const fd = c.shm_open(name_z.ptr, c.O_RDONLY, @as(c_uint, 0o600));
    if (fd < 0) return .io;
    defer closeFd(fd);
    const mapping = mapFd(fd, total_len, c.PROT_READ);
    if (mapping == c.MAP_FAILED) return .io;
    defer unmapAddr(mapping, total_len);
    const header = shmHeaderPtr(mapping);
    const header_status = validateShmHeader(desc, header);
    if (header_status != .ok) return header_status;
    if (c.shm_unlink(name_z.ptr) != 0) return .io;
    return .ok;
}

fn hostArenaFromHandle(handle: ?*KvHostArena) ?*KvHostArena {
    return handle;
}

fn hostArenaCreateInternal(
    capacity_bytes: usize,
    out_handle: *?*KvHostArena,
    out_base_ptr: *?[*]u8,
) KvStatus {
    if (capacity_bytes == 0) return .invalid_input;

    const mapping = c.mmap(
        null,
        capacity_bytes,
        c.PROT_READ | c.PROT_WRITE,
        c.MAP_PRIVATE | map_anon_flag,
        -1,
        0,
    );
    if (mapping == c.MAP_FAILED) return .io;
    errdefer _ = c.munmap(mapping, capacity_bytes);

    const arena = std.heap.c_allocator.create(KvHostArena) catch return .oom;
    arena.* = .{
        .mapping = mapping,
        .capacity_bytes = capacity_bytes,
        .next_offset = 0,
        .free_list = std.ArrayListUnmanaged(KvHostArenaRegion).empty,
    };

    out_handle.* = arena;
    out_base_ptr.* = @ptrCast(@alignCast(mapping));
    return .ok;
}

fn hostArenaDestroyInternal(handle: ?*KvHostArena) KvStatus {
    const arena = hostArenaFromHandle(handle) orelse return .invalid_input;
    arena.free_list.deinit(std.heap.c_allocator);
    if (c.munmap(arena.mapping, arena.capacity_bytes) != 0) return .io;
    std.heap.c_allocator.destroy(arena);
    return .ok;
}

fn hostArenaReservedBytesInternal(handle: ?*const KvHostArena, out_reserved_bytes: *usize) KvStatus {
    const arena = handle orelse return .invalid_input;
    var free_bytes: usize = 0;
    for (arena.free_list.items) |region| {
        free_bytes += region.len;
    }
    out_reserved_bytes.* = arena.next_offset -| free_bytes;
    return .ok;
}

fn hostArenaReserveInternal(
    handle: ?*KvHostArena,
    len: usize,
    out_region: *KvHostArenaRegion,
) KvStatus {
    const arena = hostArenaFromHandle(handle) orelse return .invalid_input;
    if (len == 0) return .invalid_input;

    var idx: usize = 0;
    while (idx < arena.free_list.items.len) : (idx += 1) {
        const free_region = arena.free_list.items[idx];
        if (free_region.len < len) continue;

        _ = arena.free_list.swapRemove(idx);
        if (free_region.len > len) {
            arena.free_list.append(
                std.heap.c_allocator,
                .{
                    .offset = free_region.offset + len,
                    .len = free_region.len - len,
                },
            ) catch return .oom;
        }
        out_region.* = .{
            .offset = free_region.offset,
            .len = len,
        };
        return .ok;
    }

    if (arena.next_offset + len > arena.capacity_bytes) return .oom;
    out_region.* = .{
        .offset = arena.next_offset,
        .len = len,
    };
    arena.next_offset += len;
    return .ok;
}

fn hostArenaReleaseInternal(handle: ?*KvHostArena, region: KvHostArenaRegion) KvStatus {
    const arena = hostArenaFromHandle(handle) orelse return .invalid_input;
    const region_end = std.math.add(usize, @intCast(region.offset), region.len) catch {
        return .invalid_input;
    };
    if (region.len == 0 or region_end > arena.capacity_bytes) return .invalid_input;
    arena.free_list.append(std.heap.c_allocator, region) catch return .oom;
    return .ok;
}

fn hostArenaResetInternal(handle: ?*KvHostArena) KvStatus {
    const arena = hostArenaFromHandle(handle) orelse return .invalid_input;
    arena.next_offset = 0;
    arena.free_list.clearRetainingCapacity();
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

pub export fn kv_native_wal_append(
    path_ptr: [*]const u8,
    path_len: usize,
    kind: u8,
    key_ptr: [*]const u8,
    key_len: usize,
    value_ptr: [*]const u8,
    value_len: usize,
) c_int {
    return @intFromEnum(walAppendInternal(
        sliceFromRaw(path_ptr, path_len),
        kind,
        sliceFromRaw(key_ptr, key_len),
        sliceFromRaw(value_ptr, value_len),
    ));
}

pub export fn kv_native_wal_replay(
    path_ptr: [*]const u8,
    path_len: usize,
    out: *KvBuffer,
) c_int {
    return @intFromEnum(walReplayInternal(sliceFromRaw(path_ptr, path_len), out));
}

pub export fn kv_native_mmap_create(
    path_ptr: [*]const u8,
    path_len: usize,
    len: usize,
    out: *KvMmapDescriptor,
) c_int {
    return @intFromEnum(mmapCreateInternal(sliceFromRaw(path_ptr, path_len), len, out));
}

pub export fn kv_native_mmap_write(
    desc: *const KvMmapDescriptor,
    offset: usize,
    bytes_ptr: [*]const u8,
    bytes_len: usize,
) c_int {
    return @intFromEnum(mmapWriteInternal(desc, offset, sliceFromRaw(bytes_ptr, bytes_len)));
}

pub export fn kv_native_mmap_read(
    desc: *const KvMmapDescriptor,
    offset: usize,
    bytes_len: usize,
    out: *KvBuffer,
) c_int {
    return @intFromEnum(mmapReadInternal(desc, offset, bytes_len, out));
}

pub export fn kv_native_shm_create(
    name_ptr: [*]const u8,
    name_len: usize,
    len: usize,
    out: *KvSharedMemoryDescriptor,
) c_int {
    return @intFromEnum(shmCreateInternal(sliceFromRaw(name_ptr, name_len), len, out));
}

pub export fn kv_native_shm_write(
    desc: *const KvSharedMemoryDescriptor,
    offset: usize,
    bytes_ptr: [*]const u8,
    bytes_len: usize,
) c_int {
    return @intFromEnum(shmWriteInternal(desc, offset, sliceFromRaw(bytes_ptr, bytes_len)));
}

pub export fn kv_native_shm_read(
    desc: *const KvSharedMemoryDescriptor,
    offset: usize,
    bytes_len: usize,
    out: *KvBuffer,
) c_int {
    return @intFromEnum(shmReadInternal(desc, offset, bytes_len, out));
}

pub export fn kv_native_shm_unlink(desc: *const KvSharedMemoryDescriptor) c_int {
    return @intFromEnum(shmUnlinkInternal(desc));
}

pub export fn kv_native_host_arena_create(
    capacity_bytes: usize,
    out_handle: *?*KvHostArena,
    out_base_ptr: *?[*]u8,
) c_int {
    return @intFromEnum(hostArenaCreateInternal(capacity_bytes, out_handle, out_base_ptr));
}

pub export fn kv_native_host_arena_destroy(handle: ?*KvHostArena) c_int {
    return @intFromEnum(hostArenaDestroyInternal(handle));
}

pub export fn kv_native_host_arena_reserved_bytes(
    handle: ?*const KvHostArena,
    out_reserved_bytes: *usize,
) c_int {
    return @intFromEnum(hostArenaReservedBytesInternal(handle, out_reserved_bytes));
}

pub export fn kv_native_host_arena_reserve(
    handle: ?*KvHostArena,
    len: usize,
    out_region: *KvHostArenaRegion,
) c_int {
    return @intFromEnum(hostArenaReserveInternal(handle, len, out_region));
}

pub export fn kv_native_host_arena_release(
    handle: ?*KvHostArena,
    region: KvHostArenaRegion,
) c_int {
    return @intFromEnum(hostArenaReleaseInternal(handle, region));
}

pub export fn kv_native_host_arena_reset(handle: ?*KvHostArena) c_int {
    return @intFromEnum(hostArenaResetInternal(handle));
}

pub export fn kv_native_buffer_free(data: ?*u8) void {
    if (data) |ptr| c.free(ptr);
}
