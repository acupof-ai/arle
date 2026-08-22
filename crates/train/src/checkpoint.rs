use std::io;
use std::path::Path;

/// Atomically refresh a `latest` symlink inside `parent` pointing at
/// `target_basename`. The symlink is relative (basename only) so the tree
/// stays rsync-safe. Refuses to overwrite a non-symlink at `latest`.
/// Atomic via `.latest.tmp` + `rename` (no missing-`latest` window).
#[cfg(unix)]
pub fn write_latest_symlink(parent: &Path, target_basename: &str) -> io::Result<()> {
    use std::os::unix::fs::symlink;

    if target_basename.contains('/') || target_basename.contains('\\') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("write_latest_symlink: target must be a basename, got {target_basename:?}"),
        ));
    }

    let link = parent.join("latest");
    match std::fs::symlink_metadata(&link) {
        Ok(meta) if meta.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "write_latest_symlink: refusing to overwrite non-symlink at {}",
                    link.display()
                ),
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    let tmp = parent.join(".latest.tmp");
    match std::fs::symlink_metadata(&tmp) {
        Ok(_) => std::fs::remove_file(&tmp)?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }

    symlink(target_basename, &tmp)?;
    if let Err(err) = std::fs::rename(&tmp, &link) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn write_latest_symlink(_parent: &Path, _target_basename: &str) -> io::Result<()> {
    Ok(())
}
